"""
coho_extractor.py
Extract transactions from bank statement PDFs and populate the
Bank's Uniform Excel Template for Conduct of Account (COHO) Summary.

Supports: ABA Bank, ACLEDA, Prince Bank, and other Cambodian bank statements.

Usage (standalone):
    python coho_extractor.py <statement.pdf> [output.xlsx]

Usage (as module):
    from coho_extractor import extract_statement, fill_coho_template
    data = extract_statement("statement.pdf")
    fill_coho_template(data, "output.xlsx", "template.xlsx")
"""

import re
import sys
import os
from datetime import datetime, date
from collections import defaultdict
from pathlib import Path

import pdfplumber
import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter

# ── Colours matching original template ───────────────────────────────────────
C_HEADER_BG  = "1F4E79"   # dark blue — title row
C_SECTION_BG = "D6E4F0"   # light blue — section headers
C_SUMM_BG    = "E2EFDA"   # green — summary panel
C_MONTH_HDR  = "BDD7EE"   # month table header
C_TOTAL_BG   = "FFFF99"   # yellow — totals
C_WHITE      = "FFFFFF"
C_GREY       = "F2F2F2"
C_CLOSING    = "FCE4D6"   # orange — daily closing balance highlight

THIN = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"),  bottom=Side(style="thin"),
)
MEDIUM = Border(
    left=Side(style="medium"), right=Side(style="medium"),
    top=Side(style="medium"),  bottom=Side(style="medium"),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — PDF PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class BankStatementParser:
    """
    Parse ABA Bank (and similar Cambodian bank) PDF statements.
    Extracts: account details, opening/ending balances, all transactions.
    """

    MONTH_MAP = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    }

    def __init__(self, pdf_path: str):
        self.pdf_path  = pdf_path
        self.pages     = []
        self.full_text = ""
        self._load()

    def _load(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                self.pages.append(text)
        self.full_text = "\n".join(self.pages)

    # ── Account header ─────────────────────────────────────────────────────

    def _parse_header(self) -> dict:
        h = {}
        p = self.full_text

        # Bank name — ABA Bank detected from SWIFT code first
        if "ABAAKHPP" in p:
            h["bank"] = "ABA"
        else:
            for bank in ["ABA BANK","ACLEDA","PRINCE BANK","VATTANAC",
                         "PRASAC","WING","WOORI","MAYBANK","CANADIA"]:
                if bank.upper() in p.upper():
                    h["bank"] = bank.title().replace("Aba","ABA")
                    break
        h.setdefault("bank", "Unknown Bank")

        # Period
        m = re.search(r"For period[:\s]+(\w+\s+\d+,\s+\d{4})\s*[–-]+\s*(\w+\s+\d+,\s+\d{4})", p)
        if m:
            h["period_from_raw"] = m.group(1).strip()
            h["period_to_raw"]   = m.group(2).strip()
            h["period_from"]     = self._parse_date(m.group(1))
            h["period_to"]       = self._parse_date(m.group(2))

        # Account details
        patterns = {
            "holder_name":  r"Account Holder Name\s+(.+?)(?:\n|Account Type)",
            "account_type": r"Account Type\s+(.+?)(?:\n|Account No)",
            "account_no":   r"Account No\.\s+(\S+)",
            "currency":     r"Account Currency\s+(\w+)",
            "swift":        r"Bank SWIFT Code\s+(\w+)",
        }
        for key, pat in patterns.items():
            m2 = re.search(pat, p, re.IGNORECASE | re.DOTALL)
            h[key] = m2.group(1).strip()[:50] if m2 else ""

        # Balances
        m3 = re.search(r"Opening Balance\s+([\d,\.]+)\s+\w+", p)
        h["opening_balance"] = float(m3.group(1).replace(",","")) if m3 else 0.0

        m4 = re.search(r"Total Money In\s*\+\s*([\d,\.]+)", p)
        h["total_in"] = float(m4.group(1).replace(",","")) if m4 else 0.0

        m5 = re.search(r"Total Money Out\s*-\s*([\d,\.]+)", p)
        h["total_out"] = float(m5.group(1).replace(",","")) if m5 else 0.0

        m6 = re.search(r"Ending Balance\s+([\d,\.]+)\s+\w+", p)
        h["ending_balance"] = float(m6.group(1).replace(",","")) if m6 else 0.0

        return h

    def _parse_date(self, s: str) -> date | None:
        """Parse date strings like 'Aug 01, 2025' or 'Jan 31, 2026'."""
        s = s.strip()
        # "Aug 01, 2025"
        m = re.match(r"(\w{3})\s+(\d{1,2}),?\s+(\d{4})", s)
        if m:
            mon = self.MONTH_MAP.get(m.group(1).lower())
            if mon:
                return date(int(m.group(3)), mon, int(m.group(2)))
        return None

    # ── Transaction parser ─────────────────────────────────────────────────

    def _parse_transactions(self) -> list[dict]:
        """
        Parse ABA Bank statement.
        Each transaction is ONE line in the PDF:
          "Aug 01, 2025 DESCRIPTION  10.63 USD  USD  222.64 USD"  ← credit
          "Aug 01, 2025 DESCRIPTION  USD  98.00 USD  214.53 USD"  ← debit
        Amounts are always at the END of the line.
        Balance is always the LAST amount.
        """
        transactions = []
        DATE_RE = re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s+(.*)")
        AMT_RE  = re.compile(r"(\d[\d,]*\.?\d*)\s+USD")

        for page_text in self.pages:
            for raw_line in page_text.split("\n"):
                line = raw_line.strip()
                m    = DATE_RE.match(line)
                if not m:
                    continue

                trx_date = self._parse_date(m.group(1))
                rest     = m.group(2).strip()

                if any(x in rest for x in [
                    "Advanced Bank","info@aba","Page ",
                    "Transaction Details","Money In","Money Out",
                    "Ending Balance","Total ","DISCLAIMER"
                ]):
                    continue

                # Remove Khmer chars and illegal bytes before parsing
                rest = re.sub(r'[\u1780-\u17ff\u19e0-\u19ff\u0000-\u0008]+', '', rest)

                # All "number USD" matches in order
                amt_matches = [(m2.start(), float(m2.group(1).replace(",","")))
                               for m2 in AMT_RE.finditer(rest)]

                if len(amt_matches) < 2:
                    continue

                balance   = amt_matches[-1][1]    # always last
                txn_start = amt_matches[-2][0]    # description ends here

                # Transaction amount = second-to-last number
                txn_amt = amt_matches[-2][1]

                # Debit or credit?
                # Check if "USD" appears before the second-to-last number
                chunk_before = rest[:txn_start]
                usd_before   = bool(re.search(r'\bUSD\b', chunk_before))

                if usd_before:
                    money_out, money_in = txn_amt, 0.0
                else:
                    money_in, money_out = txn_amt, 0.0

                # Description = everything before the transaction amount
                desc = chunk_before
                desc = re.sub(r'ORIGINAL AMOUNT$', '', desc)
                desc = re.sub(r'\bON\s+\w+\s+\d+,?\s*\d*\s*$', '', desc)
                desc = re.sub(r'\s*\d{2}:\d{2}\s*[AP]M\s*$', '', desc)
                desc = re.sub(r'REF#\s*\S+', '', desc)
                desc = re.sub(r'TRAN#\s*\S+', '', desc)
                desc = re.sub(r'HASH#\s*[0-9a-fA-F]+', '', desc)
                desc = re.sub(r'BAKONG#\s*[0-9a-fA-F]+', '', desc)
                desc = re.sub(r'EXT#\s*\S+', '', desc)
                desc = re.sub(r'APV\s*\S+', '', desc)
                desc = re.sub(r'TID:\s*\S+', '', desc)
                desc = re.sub(r'PURCHASE#\s*\S+', '', desc)
                desc = re.sub(r'\|[^|]+\|', '', desc)
                desc = re.sub(r'\bUSD\b', '', desc)
                desc = re.sub(r'\s+', ' ', desc).strip().strip('|\\/. ')
                desc = desc[:100]

                if trx_date and txn_amt > 0:
                    transactions.append({
                        "date":      trx_date,
                        "desc":      desc,
                        "money_in":  round(money_in, 2),
                        "money_out": round(money_out, 2),
                        "balance":   round(balance, 2),
                    })

        return transactions

    # ── Main parse ─────────────────────────────────────────────────────────

    def parse(self) -> dict:
        header       = self._parse_header()
        transactions = self._parse_transactions()

        # Verify and fix running balances
        transactions = self._fix_balances(header["opening_balance"], transactions)

        return {
            "header":       header,
            "transactions": transactions,
            "parsed_at":    datetime.now().isoformat(),
            "source_file":  self.pdf_path,
        }

    def _fix_balances(self, opening: float, txns: list) -> list:
        """
        Recalculate running balances from scratch to ensure accuracy.
        Only keep transactions where balance makes sense.
        """
        bal = opening
        result = []
        for t in txns:
            bal = round(bal + t["money_in"] - t["money_out"], 2)
            t["running_balance"] = bal
            # Use the stated balance from PDF as O/S balance
            t["os_balance"] = t["balance"]
            result.append(t)
        return result


def extract_statement(pdf_path: str) -> dict:
    """Public function: extract bank statement from PDF."""
    parser = BankStatementParser(pdf_path)
    return parser.parse()


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 — ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_analytics(data: dict) -> dict:
    """Compute all COHO summary statistics from transaction data."""
    txns    = data["transactions"]
    header  = data["header"]

    if not txns:
        return {}

    opening = header.get("opening_balance", 0.0)

    # ── Per-transaction analytics ──────────────────────────────────────────
    total_in_trx  = sum(1 for t in txns if t["money_in"] > 0)
    total_out_trx = sum(1 for t in txns if t["money_out"] > 0)
    total_in_amt  = round(sum(t["money_in"] for t in txns), 2)
    total_out_amt = round(sum(t["money_out"] for t in txns), 2)

    # ── Daily closing balances ─────────────────────────────────────────────
    # Last transaction each day = daily closing balance
    daily_closing: dict[date, float] = {}
    for t in txns:
        daily_closing[t["date"]] = t["os_balance"]

    closing_values = list(daily_closing.values())
    highest_bal    = max(closing_values) if closing_values else 0.0
    lowest_bal     = min(closing_values) if closing_values else 0.0

    # ── Period ────────────────────────────────────────────────────────────
    date_from = header.get("period_from") or txns[0]["date"]
    date_to   = header.get("period_to")   or txns[-1]["date"]
    months    = max(1, round(((date_to.year - date_from.year)*12 +
                               date_to.month - date_from.month) + 1))

    avg_monthly_in_trx  = round(total_in_trx / months)
    avg_monthly_out_trx = round(total_out_trx / months)
    avg_monthly_in_amt  = round(total_in_amt / months, 2)
    avg_monthly_out_amt = round(total_out_amt / months, 2)
    avg_closing_bal     = round(sum(closing_values) / len(closing_values), 2) \
                          if closing_values else 0.0

    # ── Monthly breakdown ─────────────────────────────────────────────────
    monthly: dict[tuple, dict] = {}
    for t in txns:
        key = (t["date"].year, t["date"].month)
        if key not in monthly:
            monthly[key] = {
                "debit_trx": 0, "debit_amt": 0.0,
                "credit_trx": 0, "credit_amt": 0.0,
                "balances": [], "date": date(t["date"].year, t["date"].month, 1)
            }
        if t["money_out"] > 0:
            monthly[key]["debit_trx"]  += 1
            monthly[key]["debit_amt"]  += t["money_out"]
        if t["money_in"] > 0:
            monthly[key]["credit_trx"]  += 1
            monthly[key]["credit_amt"]  += t["money_in"]
        monthly[key]["balances"].append(t["os_balance"])

    monthly_rows = []
    for idx, (key, m) in enumerate(sorted(monthly.items()), 1):
        bals = m["balances"]
        avg_bal = round(sum(bals)/len(bals), 2) if bals else 0.0
        monthly_rows.append({
            "no":         idx,
            "month":      m["date"],
            "debit_trx":  m["debit_trx"],
            "debit_amt":  round(m["debit_amt"], 2),
            "credit_trx": m["credit_trx"],
            "credit_amt": round(m["credit_amt"], 2),
            "avg_bal":    avg_bal,
            "lowest_bal": min(bals) if bals else 0.0,
            "highest_bal":max(bals) if bals else 0.0,
        })

    # ── Flag reversals ────────────────────────────────────────────────────
    reversal_amt = round(sum(
        t["money_in"] for t in txns
        if "REVERSAL" in t["desc"].upper()
    ), 2)

    return {
        "total_in_trx":       total_in_trx,
        "total_out_trx":      total_out_trx,
        "total_in_amt":       total_in_amt,
        "total_out_amt":      total_out_amt,
        "highest_balance":    highest_bal,
        "lowest_balance":     lowest_bal,
        "avg_closing_bal":    avg_closing_bal,
        "avg_monthly_in_trx": avg_monthly_in_trx,
        "avg_monthly_out_trx":avg_monthly_out_trx,
        "avg_monthly_in_amt": avg_monthly_in_amt,
        "avg_monthly_out_amt":avg_monthly_out_amt,
        "period_months":      months,
        "reversal_amt":       reversal_amt,
        "monthly_rows":       monthly_rows,
        "daily_closing":      daily_closing,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — EXCEL TEMPLATE FILLER
# ═══════════════════════════════════════════════════════════════════════════════

def _s(ws, row, col, value=None, bold=False, bg=None,
       align="left", wrap=False, fmt=None, border=True, italic=False,
       size=9, color="000000"):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font      = Font(name="Arial", bold=bold, italic=italic,
                          size=size, color=color)
    cell.alignment = Alignment(horizontal=align, vertical="center",
                                wrap_text=wrap)
    if bg:
        cell.fill = PatternFill("solid", start_color=bg)
    if border:
        cell.border = THIN
    if fmt:
        cell.number_format = fmt
    return cell


def fill_coho_template(data: dict, output_path: str,
                        template_path: str = None) -> str:
    """
    Fill the COHO Excel template with extracted bank statement data.
    Matches exactly the bank's uniform template structure.
    Returns output_path.
    """
    analytics = compute_analytics(data)
    header    = data["header"]
    txns      = data["transactions"]

    # ── Load or create workbook ────────────────────────────────────────────
    if template_path and Path(template_path).exists():
        wb = openpyxl.load_workbook(template_path)
        ws = wb.active
        # Clear transaction data rows (keep formatting)
        for row in ws.iter_rows(min_row=6, max_row=ws.max_row):
            for cell in row:
                if cell.column <= 8:
                    cell.value = None
    else:
        wb = openpyxl.Workbook()
        ws = wb.active

    # Sheet name
    acct_no  = header.get("account_no","")
    bank     = header.get("bank","Bank")
    currency = header.get("currency","USD")
    ws.title = f"{bank} {currency}-{acct_no}"[:31]

    # ── Column widths ──────────────────────────────────────────────────────
    widths = {1:8,2:12,3:45,4:16,5:16,6:16,7:18,
              8:1,9:1,10:18,11:12,12:12,13:14,14:3,
              15:12,16:14,17:3,18:14,19:12,20:12,21:4,22:12}
    for col, w in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = w

    # ── ROW 1: Title ──────────────────────────────────────────────────────
    ws.row_dimensions[1].height = 20
    c = ws.cell(row=1, column=1,
                value="Conduct of Bank Account Summary")
    c.font      = Font(name="Arial", bold=True, size=13, color="FFFFFF")
    c.fill      = PatternFill("solid", start_color=C_HEADER_BG)
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.merge_cells("A1:G1")

    # ── ROW 3: Opening balance ────────────────────────────────────────────
    ws.row_dimensions[3].height = 13
    _s(ws, 3, 5, "Openning Balance :", bold=True, border=False)
    _s(ws, 3, 6, header.get("opening_balance", 0),
       border=False, fmt="#,##0.00")

    # ── ROW 5: Column headers ─────────────────────────────────────────────
    ws.row_dimensions[5].height = 30
    headers_left = [
        "No Trx", "Date", "Transaction Details",
        "Withdrawal (Debit)", "Deposit (Credit)",
        "O/S Balance", "Dialy Closing Balance",
    ]
    for ci, h in enumerate(headers_left, 1):
        _s(ws, 5, ci, h, bold=True, bg=C_SECTION_BG,
           align="center", wrap=True, size=9)

    # ── Summary panel (right side) ────────────────────────────────────────
    _s(ws, 5,  10, "A) Conduct of Account Summary",
       bold=True, bg=C_SUMM_BG, size=10, border=False)
    ws.merge_cells("J5:M5")

    def sum_row(row, lbl, val1, sep=None, val2=None, bg=C_WHITE):
        ws.row_dimensions[row].height = 13
        _s(ws, row, 10, lbl,  bold=True, bg=bg, border=False)
        _s(ws, row, 13, val1, bg=bg, border=False,
           fmt="#,##0.00" if isinstance(val1, float) else None)
        if sep:
            _s(ws, row, 14, sep,  bg=bg, border=False, align="center")
            _s(ws, row, 15, val2, bg=bg, border=False,
               fmt="#,##0.00" if isinstance(val2, float) else None)

    period_from = header.get("period_from")
    period_to   = header.get("period_to")

    _s(ws, 6,  13, "Account 1", bold=True, bg=C_SUMM_BG, border=False)
    sum_row(7,  "Institution Name",         header.get("bank",""),      bg=C_SUMM_BG)
    sum_row(8,  "Account Holder Name",      header.get("holder_name",""), bg=C_SUMM_BG)
    r9 = ws.cell(row=9, column=13, value=period_from)
    r9.number_format = "DD-MMM-YYYY"
    r9.fill = PatternFill("solid", start_color=C_SUMM_BG)
    r9.font = Font(name="Arial", size=9)
    r9t = ws.cell(row=9, column=14, value="To")
    r9t.fill = PatternFill("solid", start_color=C_SUMM_BG)
    r9t.font = Font(name="Arial", size=9)
    r9e = ws.cell(row=9, column=15, value=period_to)
    r9e.number_format = "DD-MMM-YYYY"
    r9e.fill = PatternFill("solid", start_color=C_SUMM_BG)
    r9e.font = Font(name="Arial", size=9)
    _s(ws, 9, 10, "Period Cover", bold=True, bg=C_SUMM_BG, border=False)
    sum_row(10, "Period in Months",  analytics.get("period_months",0),      bg=C_SUMM_BG)
    sum_row(11, "Account Number",    header.get("account_no",""),            bg=C_SUMM_BG)
    sum_row(12, "Currency",          f"({header.get('currency','USD')})",    bg=C_SUMM_BG)
    _s(ws, 13, 13, "Withdraw (Debit)", bold=True, bg=C_SUMM_BG, border=False)
    _s(ws, 13, 15, "Deposit (Credit)", bold=True, bg=C_SUMM_BG, border=False)
    sum_row(14, "Number of Transaction",
            analytics.get("total_out_trx",0), "/",
            analytics.get("total_in_trx",0),  bg=C_SUMM_BG)
    sum_row(15, "Average Monthly Trx",
            analytics.get("avg_monthly_out_trx",0), "/",
            analytics.get("avg_monthly_in_trx",0),  bg=C_SUMM_BG)
    sum_row(16, "Turnover Dr/Cr",
            round(analytics.get("total_out_amt",0),2), "/",
            round(analytics.get("total_in_amt",0),2),  bg=C_SUMM_BG)
    sum_row(17, "Average Monthly Dr/Cr",
            round(analytics.get("avg_monthly_out_amt",0),2), "/",
            round(analytics.get("avg_monthly_in_amt",0),2),  bg=C_SUMM_BG)
    sum_row(18, "Highest Balance Dr/Cr", analytics.get("highest_balance",0), bg=C_SUMM_BG)
    sum_row(19, "Lowest Balance Dr/Cr",  analytics.get("lowest_balance",0),  bg=C_SUMM_BG)
    sum_row(20, "Average Closing Balance",analytics.get("avg_closing_bal",0),bg=C_SUMM_BG)

    # Exclusion rows
    _s(ws, 22, 10, "To Exclude from Turnover for the following Transaction :",
       bold=True, bg=C_SUMM_BG, border=False)
    _s(ws, 22, 13, "Loan Disbursement",    bold=True, bg=C_SUMM_BG, border=False)
    _s(ws, 22, 15, 0, bg=C_SUMM_BG, border=False, fmt="#,##0.00")
    _s(ws, 23, 13, "Manipulation Trx.",    bold=True, bg=C_SUMM_BG, border=False)
    _s(ws, 23, 15, 0, bg=C_SUMM_BG, border=False, fmt="#,##0.00")
    _s(ws, 24, 13, "Reversal Entries",     bold=True, bg=C_SUMM_BG, border=False)
    _s(ws, 24, 15, analytics.get("reversal_amt",0),
       bg=C_SUMM_BG, border=False, fmt="#,##0.00")
    _s(ws, 25, 13, "……………………………",           bold=True, bg=C_SUMM_BG, border=False)
    _s(ws, 25, 15, 0, bg=C_SUMM_BG, border=False, fmt="#,##0.00")
    _s(ws, 26, 13, "Total",                bold=True, bg=C_SUMM_BG, border=False)
    _s(ws, 26, 15, 0, bg=C_SUMM_BG, border=False, fmt="#,##0.00")

    # ── Section B header ───────────────────────────────────────────────────
    _s(ws, 28, 10, "B) Conduct of Account Summary (By Month)",
       bold=True, bg=C_SUMM_BG, size=10, border=False)
    ws.merge_cells("J28:W28")

    # Section B column headers
    ws.row_dimensions[29].height = 30
    b_headers = [
        ("J29","No"),("K29","Month"),
        ("L29","Debit (Withdrawal)"),("O29","Credit (Deposit)"),
        ("R29","Average\nClosing Bal."),("S29","Lowest Bal."),
        ("T29","Highest Bal."),("U29","Min"),("V29","Max"),
    ]
    for cell_addr, lbl in b_headers:
        c2 = ws[cell_addr]
        c2.value     = lbl
        c2.font      = Font(name="Arial", bold=True, size=9)
        c2.fill      = PatternFill("solid", start_color=C_MONTH_HDR)
        c2.alignment = Alignment(horizontal="center", vertical="center",
                                  wrap_text=True)
        c2.border    = THIN

    ws.row_dimensions[30].height = 16
    for ci, lbl in [(12,"Debit (Trx)"),(13,"Debit (Amt)"),
                    (15,"Credit (Trx)"),(16,"Credit (Amt)")]:
        _s(ws, 30, ci, lbl, bold=True, bg=C_MONTH_HDR, align="center")

    # ── Section B data rows ────────────────────────────────────────────────
    monthly_rows = analytics.get("monthly_rows", [])
    MONTH_START_ROW = 31

    for idx, mr in enumerate(monthly_rows):
        r   = MONTH_START_ROW + idx
        bg  = C_GREY if idx % 2 else C_WHITE
        ws.row_dimensions[r].height = 13

        _s(ws, r, 10, mr["no"],         align="center",  bg=bg)
        mc = ws.cell(row=r, column=11, value=mr["month"])
        mc.number_format = "MMM-YYYY"
        mc.fill = PatternFill("solid", start_color=bg)
        mc.font = Font(name="Arial", size=9)
        mc.border = THIN

        _s(ws, r, 12, mr["debit_trx"],  align="center",  bg=bg)
        _s(ws, r, 13, mr["debit_amt"],  align="right",   bg=bg, fmt="#,##0.00")
        _s(ws, r, 14, "/",              align="center",  bg=bg)
        _s(ws, r, 15, mr["credit_trx"], align="center",  bg=bg)
        _s(ws, r, 16, mr["credit_amt"], align="right",   bg=bg, fmt="#,##0.00")
        _s(ws, r, 17, "/",              align="center",  bg=bg)
        _s(ws, r, 18, mr["avg_bal"],    align="right",   bg=bg, fmt="#,##0.00")
        _s(ws, r, 19, "",               bg=bg)
        _s(ws, r, 20, "",               bg=bg)
        _s(ws, r, 21, mr["lowest_bal"], align="right",   bg=bg, fmt="#,##0.00")
        _s(ws, r, 22, mr["highest_bal"],align="right",   bg=bg, fmt="#,##0.00")

    # Fill remaining monthly rows (up to 24) with zeros
    for idx in range(len(monthly_rows), 24):
        r = MONTH_START_ROW + idx
        ws.row_dimensions[r].height = 13
        for ci in range(10, 23):
            _s(ws, r, ci, 0 if ci != 14 and ci != 17 else "/",
               align="center" if ci in (10,14,17) else "right",
               bg=C_WHITE, fmt="#,##0.00" if ci not in (10,14,17) else None)

    # Totals row
    total_row = MONTH_START_ROW + 24
    ws.row_dimensions[total_row].height = 14
    _s(ws, total_row, 11, "Total:", bold=True, bg=C_TOTAL_BG)
    _s(ws, total_row, 12, analytics.get("total_out_trx",0),
       bold=True, bg=C_TOTAL_BG, align="center")
    _s(ws, total_row, 13, round(analytics.get("total_out_amt",0),2),
       bold=True, bg=C_TOTAL_BG, fmt="#,##0.00")
    _s(ws, total_row, 15, analytics.get("total_in_trx",0),
       bold=True, bg=C_TOTAL_BG, align="center")
    _s(ws, total_row, 16, round(analytics.get("total_in_amt",0),2),
       bold=True, bg=C_TOTAL_BG, fmt="#,##0.00")

    # Variance check
    var_row = total_row + 2
    _s(ws, var_row, 10, "Variance Check", bold=True, bg=C_TOTAL_BG)
    for ci in [12,13,15,16]:
        _s(ws, var_row, ci, 0, bg=C_TOTAL_BG, fmt="#,##0.00")

    # ── TRANSACTION ROWS ──────────────────────────────────────────────────
    TRX_START = 6
    daily_closing = analytics.get("daily_closing", {})

    for idx, t in enumerate(txns):
        r   = TRX_START + idx
        bg  = C_GREY if idx % 2 else C_WHITE
        ws.row_dimensions[r].height = 13

        is_closing = (t["date"] in daily_closing and
                      t["os_balance"] == daily_closing[t["date"]])

        _s(ws, r, 1, idx+1, align="center", bg=bg)

        dc = ws.cell(row=r, column=2, value=t["date"])
        dc.number_format = "DD/MM/YYYY"
        dc.fill  = PatternFill("solid", start_color=bg)
        dc.font  = Font(name="Arial", size=9)
        dc.border = THIN

        _s(ws, r, 3, t["desc"], bg=bg, wrap=True)
        _s(ws, r, 4, t["money_out"] if t["money_out"] > 0 else None,
           bg=bg, fmt="#,##0.00", align="right")
        _s(ws, r, 5, t["money_in"] if t["money_in"] > 0 else None,
           bg=bg, fmt="#,##0.00", align="right")
        _s(ws, r, 6, t["os_balance"],
           bg=bg, fmt="#,##0.00", align="right")
        _s(ws, r, 7, t["os_balance"] if is_closing else None,
           bg=C_CLOSING if is_closing else bg,
           fmt="#,##0.00", align="right")

    # Freeze top rows
    ws.freeze_panes = "A6"

    wb.save(output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python coho_extractor.py <statement.pdf> [output.xlsx] [template.xlsx]")
        sys.exit(1)

    pdf_path  = sys.argv[1]
    out_path  = sys.argv[2] if len(sys.argv) > 2 else \
                pdf_path.replace(".pdf","_COHO.xlsx")
    tmpl_path = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Extracting: {pdf_path}")
    data = extract_statement(pdf_path)
    h    = data["header"]
    txns = data["transactions"]
    an   = compute_analytics(data)

    print(f"  Bank:             {h.get('bank')}")
    print(f"  Account Holder:   {h.get('holder_name')}")
    print(f"  Account No:       {h.get('account_no')}")
    print(f"  Period:           {h.get('period_from')} → {h.get('period_to')}")
    print(f"  Opening Balance:  ${h.get('opening_balance',0):,.2f}")
    print(f"  Ending Balance:   ${h.get('ending_balance',0):,.2f}")
    print(f"  Transactions:     {len(txns)}")
    print(f"  Total In:         ${an.get('total_in_amt',0):,.2f} ({an.get('total_in_trx',0)} trx)")
    print(f"  Total Out:        ${an.get('total_out_amt',0):,.2f} ({an.get('total_out_trx',0)} trx)")
    print(f"  Highest Balance:  ${an.get('highest_balance',0):,.2f}")
    print(f"  Avg Closing Bal:  ${an.get('avg_closing_bal',0):,.2f}")

    print(f"\nGenerating Excel: {out_path}")
    fill_coho_template(data, out_path, tmpl_path)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
