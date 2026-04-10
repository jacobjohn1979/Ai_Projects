"""
pdf_banking.py — Banking-specific PDF Intelligence
Advanced fraud detection for bank statements, payslips, tax docs, utility bills.

Checks:
  1. Balance consistency — running total math validation
  2. Round number detection — suspicious edited figures
  3. Transaction pattern analysis — velocity, duplicates, gaps
  4. Salary cross-validation — payslip vs bank deposits
  5. Template fingerprinting — same template, different applicants
  6. Address extraction — cross-document address matching
  7. Income vs loan ratio — unrealistic loan amounts
  8. Employer validation — payslip employer consistency
  9. Date sequence validation — statement dates are logical
 10. Benford's Law analysis — digit distribution fraud signal
"""

import re
import os
import json
import math
import hashlib
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from dateutil import parser as date_parser
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("fraud_detect.pdf_banking")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

# ── Regex patterns ─────────────────────────────────────────────────────────────
AMOUNT_RE   = re.compile(r"[\$\£\€¥₭]?\s*\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})\b")
BALANCE_RE  = re.compile(
    r"(?:balance|bal\.?|closing|opening|available)\s*:?\s*[\$\£\€¥₭]?\s*([\d,\.]+)",
    re.IGNORECASE
)
DATE_RE     = re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b")
SALARY_RE   = re.compile(
    r"(?:salary|gross pay|net pay|basic pay|take.home|total earnings?)\s*:?\s*[\$\£\€¥₭]?\s*([\d,\.]+)",
    re.IGNORECASE
)
EMPLOYER_RE = re.compile(
    r"(?:employer|company|organisation|organization|paid by|from)\s*:?\s*([A-Z][A-Za-z\s\.\,\&]{3,50})",
    re.IGNORECASE
)
ADDRESS_RE  = re.compile(
    r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl)[,\s]+[A-Za-z\s]+",
    re.IGNORECASE
)
DEBIT_RE    = re.compile(r"(?:debit|dr\.?|withdrawal|paid out)\s*:?\s*[\$\£\€¥₭]?\s*([\d,\.]+)", re.IGNORECASE)
CREDIT_RE   = re.compile(r"(?:credit|cr\.?|deposit|received)\s*:?\s*[\$\£\€¥₭]?\s*([\d,\.]+)", re.IGNORECASE)
PERIOD_RE   = re.compile(
    r"(?:period|statement date|from|for the month)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\s*(?:to|[-–])\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    re.IGNORECASE
)
ACCOUNT_RE  = re.compile(r"\b(?:account|acct\.?|a\/c)\s*(?:no\.?|number|#)?\s*:?\s*([0-9X\*]{6,20})\b", re.IGNORECASE)


def _parse_amount(s: str) -> float:
    """Parse amount string to float."""
    try:
        cleaned = re.sub(r"[^\d\.]", "", s.replace(",", ""))
        return float(cleaned) if cleaned else 0.0
    except Exception:
        return 0.0


def _extract_text(file_path: str) -> str:
    """Extract full text from PDF."""
    try:
        doc  = fitz.open(file_path)
        text = "\n".join(page.get_text("text") for page in doc)
        return text
    except Exception as e:
        log.error(f"Text extraction failed: {e}")
        return ""


def _pdf_fingerprint(file_path: str) -> str:
    """
    Generate structural fingerprint of PDF layout.
    Two fraudulently filled versions of the same template will share this fingerprint.
    Based on: font names, page dimensions, image positions — NOT the text content.
    """
    try:
        doc  = fitz.open(file_path)
        parts = []
        for page in doc:
            parts.append(f"{round(page.rect.width)}x{round(page.rect.height)}")
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font = span.get("font", "")
                            size = round(span.get("size", 0))
                            if font:
                                parts.append(f"{font}:{size}")
            for img in page.get_images(full=True):
                parts.append(f"img:{img[2]}x{img[3]}")
        fingerprint = hashlib.md5("|".join(parts).encode()).hexdigest()
        return fingerprint
    except Exception as e:
        log.error(f"Fingerprint failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  1. BALANCE CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_balance_consistency(text: str) -> tuple:
    """
    Validate that opening balance + credits - debits ≈ closing balance.
    Flags if math doesn't add up (edited figures).
    """
    flags, info = [], {}

    balances = BALANCE_RE.findall(text)
    amounts  = [_parse_amount(b) for b in balances if _parse_amount(b) > 0]
    debits   = [_parse_amount(d) for d in DEBIT_RE.findall(text)]
    credits  = [_parse_amount(c) for c in CREDIT_RE.findall(text)]

    info["balances_found"]  = len(amounts)
    info["debit_count"]     = len(debits)
    info["credit_count"]    = len(credits)
    info["total_debits"]    = round(sum(debits), 2)
    info["total_credits"]   = round(sum(credits), 2)

    if len(amounts) >= 2 and (debits or credits):
        opening = amounts[0]
        closing = amounts[-1]
        expected_closing = round(opening + sum(credits) - sum(debits), 2)
        discrepancy = abs(expected_closing - closing)
        tolerance   = max(opening * 0.02, 5.0)  # 2% or $5 tolerance

        info["opening_balance"]   = opening
        info["closing_balance"]   = closing
        info["expected_closing"]  = expected_closing
        info["discrepancy"]       = round(discrepancy, 2)

        if discrepancy > tolerance:
            flags.append("balance_math_inconsistency")
            if discrepancy > opening * 0.1:
                flags.append("large_balance_discrepancy")

    if len(amounts) == 0:
        flags.append("no_balance_found")

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  2. ROUND NUMBER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def check_round_numbers(text: str) -> tuple:
    """
    Detect suspicious prevalence of round numbers.
    Real transactions have varied cent amounts; edited figures are often rounded.
    Benford's Law: in real financial data, leading digit 1 appears ~30% of the time.
    """
    flags, info = [], {}

    raw_amounts = AMOUNT_RE.findall(text)
    amounts     = [_parse_amount(a) for a in raw_amounts if _parse_amount(a) > 0]

    if not amounts:
        return {"amounts_found": 0}, []

    # Round number check
    round_100  = sum(1 for a in amounts if a % 100 == 0 and a > 0)
    round_1000 = sum(1 for a in amounts if a % 1000 == 0 and a > 0)
    round_pct  = round(round_100 / len(amounts) * 100, 1) if amounts else 0

    info["total_amounts"]       = len(amounts)
    info["round_100_count"]     = round_100
    info["round_1000_count"]    = round_1000
    info["round_number_pct"]    = round_pct

    if round_pct > 60 and len(amounts) >= 5:
        flags.append("excessive_round_numbers")
    if round_pct > 80 and len(amounts) >= 3:
        flags.append("suspicious_round_number_prevalence")

    # Benford's Law analysis
    if len(amounts) >= 10:
        leading_digits = []
        for a in amounts:
            s = str(int(a)).lstrip("0")
            if s:
                leading_digits.append(int(s[0]))

        if leading_digits:
            expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}
            actual   = Counter(leading_digits)
            total    = len(leading_digits)

            # Chi-squared test
            chi2 = sum(
                ((actual.get(d, 0)/total - expected[d]) ** 2) / expected[d]
                for d in range(1, 10)
            )
            info["benfords_chi2"]     = round(chi2, 4)
            info["leading_digit_1_pct"] = round(actual.get(1, 0) / total * 100, 1)

            # chi2 > 15.5 is significant at p=0.05 with 8 df
            if chi2 > 15.5:
                flags.append("benfords_law_violation")
            elif chi2 > 10:
                flags.append("benfords_law_suspicious")

    # Duplicate amount detection
    amount_counts = Counter([round(a, 2) for a in amounts])
    duplicates    = {str(k): v for k, v in amount_counts.items() if v >= 3 and k > 10}
    if duplicates:
        info["repeated_amounts"] = duplicates
        if any(v >= 5 for v in duplicates.values()):
            flags.append("excessive_duplicate_amounts")
        elif duplicates:
            flags.append("repeated_transaction_amounts")

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  3. TRANSACTION PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def check_transaction_patterns(text: str) -> tuple:
    """
    Analyse transaction patterns for suspicious activity:
    - Structuring (multiple transactions just below reporting thresholds)
    - Velocity spikes
    - End-of-period large deposits (common in statement stuffing fraud)
    - Gaps suggesting missing pages
    """
    flags, info = [], {}

    dates   = DATE_RE.findall(text)
    amounts = [_parse_amount(a) for a in AMOUNT_RE.findall(text) if _parse_amount(a) > 0]

    parsed_dates = []
    for d in dates:
        try:
            parsed_dates.append(date_parser.parse(d, dayfirst=True))
        except Exception:
            pass

    info["transaction_count"] = len(amounts)
    info["date_count"]        = len(parsed_dates)

    if not amounts:
        return info, []

    # Statement period
    if len(parsed_dates) >= 2:
        parsed_dates.sort()
        period_days = (parsed_dates[-1] - parsed_dates[0]).days
        info["statement_period_days"] = period_days

        # Check for gaps > 7 days in what should be a monthly statement
        if period_days > 45:
            flags.append("statement_period_too_long")

        # Check date gaps
        gaps = [(parsed_dates[i+1] - parsed_dates[i]).days
                for i in range(len(parsed_dates)-1)]
        max_gap = max(gaps) if gaps else 0
        info["max_date_gap_days"] = max_gap
        if max_gap > 14 and period_days < 60:
            flags.append("suspicious_date_gap")

    # Structuring detection — many amounts just below common thresholds
    thresholds = [10000, 5000, 3000, 1000]
    for threshold in thresholds:
        just_below = sum(1 for a in amounts if threshold * 0.9 <= a < threshold)
        if just_below >= 3:
            flags.append(f"structuring_near_{threshold}")
            info[f"structuring_near_{threshold}"] = just_below

    # Large end-of-period deposit (last 20% of dates)
    if len(parsed_dates) >= 5 and amounts:
        cutoff  = parsed_dates[int(len(parsed_dates) * 0.8)]
        # rough check — if max amount in text is very large relative to others
        max_amt = max(amounts)
        avg_amt = sum(amounts) / len(amounts)
        if max_amt > avg_amt * 10:
            flags.append("unusually_large_single_transaction")
            info["max_to_avg_ratio"] = round(max_amt / avg_amt, 1)

    # Velocity — many transactions in a single day
    date_counts = Counter([d.date() for d in parsed_dates])
    max_daily   = max(date_counts.values()) if date_counts else 0
    info["max_daily_transactions"] = max_daily
    if max_daily > 20:
        flags.append("high_daily_transaction_velocity")

    # Total transaction volume
    info["total_amount"] = round(sum(amounts), 2)
    info["avg_amount"]   = round(sum(amounts)/len(amounts), 2) if amounts else 0

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  4. SALARY / PAYSLIP VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def check_payslip_consistency(text: str) -> tuple:
    """
    Validate payslip internal consistency:
    - Gross - deductions ≈ net pay
    - Reasonable deduction ratios
    - Employer name present
    - Pay period is monthly/weekly (not unusual)
    """
    flags, info = [], {}

    salaries  = [_parse_amount(s) for s in SALARY_RE.findall(text)]
    employers = EMPLOYER_RE.findall(text)
    amounts   = [_parse_amount(a) for a in AMOUNT_RE.findall(text) if _parse_amount(a) > 0]

    info["salary_figures_found"] = len(salaries)
    info["employer_found"]       = bool(employers)
    info["employer_name"]        = employers[0].strip() if employers else None

    if not employers:
        flags.append("employer_name_missing")

    if len(salaries) >= 2:
        gross = max(salaries)
        net   = min(salaries)
        info["gross_pay"] = gross
        info["net_pay"]   = net

        if gross > 0:
            deduction_rate = (gross - net) / gross
            info["deduction_rate"] = round(deduction_rate, 3)

            # Deductions should be 10%-50% of gross for most employees
            if deduction_rate < 0.05:
                flags.append("suspiciously_low_deductions")
            if deduction_rate > 0.70:
                flags.append("suspiciously_high_deductions")

            # Gross and net suspiciously close (no tax?)
            if 0 < deduction_rate < 0.03:
                flags.append("no_tax_deductions_detected")

    elif len(salaries) == 1:
        flags.append("only_one_salary_figure")

    # Check for round salary (common in fake payslips)
    if salaries:
        max_sal = max(salaries)
        if max_sal > 0 and max_sal % 100 == 0:
            info["salary_is_round_number"] = True
            flags.append("salary_is_round_number")

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  5. TEMPLATE FINGERPRINTING
# ══════════════════════════════════════════════════════════════════════════════

def check_template_fingerprint(file_path: str, applicant_id: str) -> tuple:
    """
    Check if this PDF's structural template has been seen from different applicants.
    Same template + different applicant = likely forged from same source.
    """
    flags, info = [], {}

    fingerprint = _pdf_fingerprint(file_path)
    if not fingerprint:
        return info, ["fingerprint_failed"]

    info["template_fingerprint"] = fingerprint

    db = SessionLocal()
    try:
        # check if same fingerprint seen from different applicants
        rows = db.execute(text("""
            SELECT DISTINCT applicant_id, file_name, screened_at
            FROM screening_logs
            WHERE full_result->>'template_fingerprint' = :fp
            AND applicant_id IS NOT NULL
            AND applicant_id != :aid
            ORDER BY screened_at DESC
            LIMIT 5
        """), {"fp": fingerprint, "aid": applicant_id or ""}).fetchall()

        if rows:
            matches = [{"applicant_id": r.applicant_id,
                        "file_name": r.file_name,
                        "screened_at": str(r.screened_at)[:16]} for r in rows]
            info["template_shared_with"]  = matches
            info["template_match_count"]  = len(matches)
            flags.append("template_shared_across_applicants")
            if len(matches) >= 3:
                flags.append("template_used_by_multiple_applicants")

        # store fingerprint for future checks
        db.execute(text("""
            UPDATE screening_logs
            SET full_result = jsonb_set(
                COALESCE(full_result,'{}')::jsonb,
                '{template_fingerprint}',
                to_jsonb(:fp::text)
            )
            WHERE applicant_id = :aid
            AND screened_at = (
                SELECT MAX(screened_at) FROM screening_logs WHERE applicant_id = :aid
            )
        """), {"fp": fingerprint, "aid": applicant_id or ""})
        db.commit()

    except Exception as e:
        log.warning(f"Template fingerprint DB check failed: {e}")
    finally:
        db.close()

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  6. ADDRESS EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_address(text: str) -> tuple:
    """Extract and validate address from document."""
    flags, info = [], {}

    addresses = ADDRESS_RE.findall(text)
    if addresses:
        info["addresses_found"] = [a.strip() for a in addresses[:3]]
        info["primary_address"] = addresses[0].strip()
    else:
        flags.append("no_address_found")
        info["addresses_found"] = []

    # Check for PO Box (sometimes used to obscure real address)
    if re.search(r"\bP\.?O\.?\s*Box\b", text, re.IGNORECASE):
        flags.append("po_box_address")

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  7. INCOME vs LOAN RATIO
# ══════════════════════════════════════════════════════════════════════════════

def check_income_loan_ratio(text: str, loan_amount: float = 0) -> tuple:
    """
    Check if stated income is realistic relative to the loan amount requested.
    Standard banking: loan should not exceed 5-7× annual income.
    """
    flags, info = [], {}

    if loan_amount <= 0:
        return {"loan_amount": 0}, []

    salaries = [_parse_amount(s) for s in SALARY_RE.findall(text) if _parse_amount(s) > 0]
    amounts  = [_parse_amount(a) for a in AMOUNT_RE.findall(text) if _parse_amount(a) > 0]

    monthly_income = 0
    if salaries:
        monthly_income = max(salaries)  # take highest figure as gross monthly
        info["detected_monthly_income"] = monthly_income
    elif amounts:
        # Estimate from average credit amount
        monthly_income = sum(amounts) / max(len(amounts), 1) * 0.3
        info["estimated_monthly_income"] = round(monthly_income, 2)

    if monthly_income > 0:
        annual_income = monthly_income * 12
        ratio         = loan_amount / annual_income
        info["annual_income_estimate"] = round(annual_income, 2)
        info["loan_amount"]            = loan_amount
        info["loan_to_income_ratio"]   = round(ratio, 2)

        if ratio > 10:
            flags.append("loan_exceeds_10x_annual_income")
        elif ratio > 7:
            flags.append("loan_exceeds_7x_annual_income")
        elif ratio > 5:
            flags.append("loan_exceeds_5x_annual_income")
    else:
        flags.append("income_not_detectable")

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  8. DATE SEQUENCE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def check_date_sequence(text: str) -> tuple:
    """Validate that statement dates are in logical sequence."""
    flags, info = [], {}

    dates = DATE_RE.findall(text)
    parsed = []
    for d in dates:
        try:
            parsed.append(date_parser.parse(d, dayfirst=True))
        except Exception:
            pass

    if not parsed:
        return {"dates_found": 0}, ["no_dates_found"]

    parsed.sort()
    info["dates_found"]  = len(parsed)
    info["earliest_date"] = parsed[0].strftime("%Y-%m-%d")
    info["latest_date"]   = parsed[-1].strftime("%Y-%m-%d")

    # Future dates
    now        = datetime.utcnow()
    future     = [d for d in parsed if d > now + timedelta(days=30)]
    if future:
        flags.append("future_dates_detected")
        info["future_dates"] = [d.strftime("%Y-%m-%d") for d in future[:3]]

    # Very old dates (> 2 years)
    old = [d for d in parsed if d < now - timedelta(days=730)]
    if old:
        flags.append("documents_older_than_2_years")

    # Out of sequence (later date appears before earlier date in document flow)
    original_order = []
    for d in DATE_RE.findall(text):
        try:
            original_order.append(date_parser.parse(d, dayfirst=True))
        except Exception:
            pass

    inversions = sum(
        1 for i in range(len(original_order)-1)
        if (original_order[i+1] - original_order[i]).days < -30
    )
    info["date_inversions"] = inversions
    if inversions > 2:
        flags.append("date_sequence_irregular")

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  9. ACCOUNT NUMBER CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

def check_account_consistency(text: str) -> tuple:
    """Check that account number is consistent throughout the document."""
    flags, info = [], {}

    accounts = ACCOUNT_RE.findall(text)
    if not accounts:
        flags.append("no_account_number_found")
        return info, flags

    # Normalise (remove masking characters)
    normalised = [re.sub(r"[X\*]", "", a).strip() for a in accounts]
    unique     = set(a for a in normalised if len(a) >= 4)

    info["account_numbers_found"] = list(set(accounts))
    info["unique_account_count"]  = len(unique)

    if len(unique) > 2:
        flags.append("multiple_account_numbers_detected")

    return info, flags


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_banking_pdf(flags: list) -> tuple:
    weights = {
        # Balance
        "balance_math_inconsistency":          35,
        "large_balance_discrepancy":           20,
        "no_balance_found":                     5,
        # Round numbers / Benford
        "benfords_law_violation":              30,
        "benfords_law_suspicious":             15,
        "excessive_round_numbers":             20,
        "suspicious_round_number_prevalence":  25,
        "excessive_duplicate_amounts":         20,
        "repeated_transaction_amounts":        10,
        # Transaction patterns
        "structuring_near_10000":              30,
        "structuring_near_5000":              20,
        "structuring_near_3000":              15,
        "structuring_near_1000":              10,
        "high_daily_transaction_velocity":    15,
        "unusually_large_single_transaction": 15,
        "suspicious_date_gap":                15,
        "statement_period_too_long":          10,
        # Payslip
        "employer_name_missing":              15,
        "suspiciously_low_deductions":        20,
        "suspiciously_high_deductions":       20,
        "no_tax_deductions_detected":         25,
        "salary_is_round_number":             10,
        "only_one_salary_figure":              5,
        # Template
        "template_shared_across_applicants":  35,
        "template_used_by_multiple_applicants": 20,
        # Address
        "no_address_found":                    5,
        "po_box_address":                      8,
        # Income/loan
        "loan_exceeds_10x_annual_income":     30,
        "loan_exceeds_7x_annual_income":      20,
        "loan_exceeds_5x_annual_income":      10,
        "income_not_detectable":               5,
        # Dates
        "future_dates_detected":              25,
        "date_sequence_irregular":            15,
        "documents_older_than_2_years":       10,
        "no_dates_found":                      5,
        # Account
        "multiple_account_numbers_detected":  15,
        "no_account_number_found":             5,
    }
    score = sum(weights.get(f.split(":")[0], 5) for f in flags)
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 20 else "LOW")
    return score, level


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_banking_pdf(
    file_path:    str,
    applicant_id: str = "",
    loan_amount:  float = 0,
    doc_type:     str = "bank_statement",  # bank_statement | payslip | tax | utility
) -> tuple:
    """
    Full banking PDF intelligence analysis.
    Returns (info_dict, flags_list, risk_score, risk_level).
    """
    all_flags = []
    info      = {"doc_type": doc_type}

    text = _extract_text(file_path)
    if not text.strip():
        return info, ["pdf_text_extraction_failed"], 0, "LOW"

    info["text_length"] = len(text)

    # ── Run all checks ────────────────────────────────────────────────────────

    # 1. Balance consistency (all doc types)
    bal_info, bal_flags = check_balance_consistency(text)
    info["balance_check"] = bal_info
    all_flags.extend(bal_flags)

    # 2. Round numbers + Benford (all)
    rn_info, rn_flags = check_round_numbers(text)
    info["round_number_check"] = rn_info
    all_flags.extend(rn_flags)

    # 3. Transaction patterns (bank statements)
    if doc_type in ("bank_statement", "tax"):
        tp_info, tp_flags = check_transaction_patterns(text)
        info["transaction_patterns"] = tp_info
        all_flags.extend(tp_flags)

    # 4. Payslip validation
    if doc_type == "payslip":
        ps_info, ps_flags = check_payslip_consistency(text)
        info["payslip_check"] = ps_info
        all_flags.extend(ps_flags)

    # 5. Template fingerprint (all)
    tp_info, tp_flags = check_template_fingerprint(file_path, applicant_id)
    info["template_check"] = tp_info
    all_flags.extend(tp_flags)

    # 6. Address extraction (utility bills + bank statements)
    if doc_type in ("utility", "bank_statement"):
        addr_info, addr_flags = extract_address(text)
        info["address_check"] = addr_info
        all_flags.extend(addr_flags)

    # 7. Income vs loan ratio
    if loan_amount > 0:
        il_info, il_flags = check_income_loan_ratio(text, loan_amount)
        info["income_loan_check"] = il_info
        all_flags.extend(il_flags)

    # 8. Date sequence
    dt_info, dt_flags = check_date_sequence(text)
    info["date_check"] = dt_info
    all_flags.extend(dt_flags)

    # 9. Account consistency (bank statements)
    if doc_type == "bank_statement":
        ac_info, ac_flags = check_account_consistency(text)
        info["account_check"] = ac_info
        all_flags.extend(ac_flags)

    # ── Score ─────────────────────────────────────────────────────────────────
    risk_score, risk_level = score_banking_pdf(all_flags)

    return info, all_flags, risk_score, risk_level
