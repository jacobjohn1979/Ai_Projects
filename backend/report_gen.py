"""
report_gen.py — PDF Compliance Report Generator
Generates regulator-ready PDF reports per screening case.
Uses reportlab (pure Python — no wkhtmltopdf needed).
"""
import os
import io
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("fraud_detect.report_gen")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, white, grey
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    log.warning("reportlab not installed — PDF reports unavailable. Run: pip install reportlab")

RISK_COLORS = {
    "HIGH":   "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW":    "#22c55e",
}
ACTION_LABELS = {
    "HIGH":   "REJECT",
    "MEDIUM": "REVIEW",
    "LOW":    "PASS",
}


def generate_pdf_report(record: dict) -> bytes | None:
    """
    Generate a PDF compliance report for a screening record.
    Returns bytes of the PDF or None if reportlab not installed.
    """
    if not REPORTLAB_OK:
        return None

    result = record.get("full_result") or {}
    if isinstance(result, str):
        try: result = json.loads(result)
        except: result = {}

    level    = record.get("risk_level", "—")
    score    = record.get("risk_score", 0)
    action   = ACTION_LABELS.get(level, "—")
    flags    = record.get("flags") or []
    if isinstance(flags, str):
        try: flags = json.loads(flags)
        except: flags = []

    screened  = str(record.get("screened_at", ""))[:19].replace("T", " ")
    filename  = record.get("file_name", "—")
    applicant = record.get("applicant_id") or "—"
    doc_type  = str(record.get("doc_type", "—")).upper()
    sha256    = str(record.get("file_sha256") or "—")[:32] + "…"

    risk_color   = HexColor(RISK_COLORS.get(level, "#94a3b8"))
    fi           = result.get("field_info", {})
    ela          = result.get("ela", {})
    holo         = result.get("hologram", {})
    tmpl         = result.get("template_match", {})
    ml           = result.get("ml_inference", {})
    face         = result.get("face_match", {})
    staff_dec    = result.get("staff_decision", {})

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles  = getSampleStyleSheet()
    c_dark  = HexColor("#0f172a")
    c_gray  = HexColor("#64748b")
    c_light = HexColor("#f8fafc")
    c_border = HexColor("#e2e8f0")

    def style(name="Normal", **kwargs):
        s = ParagraphStyle(name, parent=styles["Normal"], **kwargs)
        return s

    h1  = style("h1",  fontSize=18, textColor=c_dark,  spaceAfter=4,  fontName="Helvetica-Bold")
    h2  = style("h2",  fontSize=11, textColor=c_dark,  spaceBefore=12, spaceAfter=4, fontName="Helvetica-Bold")
    h3  = style("h3",  fontSize=9,  textColor=c_gray,  spaceBefore=8,  spaceAfter=2, fontName="Helvetica-Bold")
    body = style("body", fontSize=9,  textColor=c_dark,  spaceAfter=2)
    small = style("sm", fontSize=8,  textColor=c_gray)
    code  = style("cd", fontSize=8,  textColor=c_dark,  fontName="Courier")

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("KINGDOM OF CAMBODIA", style("hdr", fontSize=8, textColor=c_gray)),
        Paragraph("DOCUMENT SCREENING REPORT", style("hdr2", fontSize=14, textColor=c_dark, fontName="Helvetica-Bold", alignment=TA_CENTER)),
        Paragraph(f"Report ID: {record.get('id','—')}", style("hdr3", fontSize=8, textColor=c_gray, alignment=TA_RIGHT)),
    ]]
    header_table = Table(header_data, colWidths=[4*cm, 9*cm, 4*cm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LINEBELOW", (0,0), (-1,-1), 0.5, c_border),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Risk banner ───────────────────────────────────────────────────────────
    banner_data = [[
        Paragraph(f"{level} RISK", style("risk", fontSize=16, textColor=white,
                                          fontName="Helvetica-Bold")),
        Paragraph(f"Score: {score} | Action: {action}",
                  style("score", fontSize=10, textColor=white, alignment=TA_RIGHT)),
    ]]
    banner = Table(banner_data, colWidths=[8*cm, 9*cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), risk_color),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), 4),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.4*cm))

    # ── Document details ──────────────────────────────────────────────────────
    story.append(Paragraph("1. DOCUMENT INFORMATION", h2))
    det_data = [
        ["File Name",     filename,      "Document Type",  doc_type],
        ["Applicant ID",  applicant,     "Screened At",    screened],
        ["ID Number",     fi.get("id_number") or "—",
         "Date of Birth", fi.get("dob") or "—"],
        ["Expiry Date",   fi.get("expiry_date") or "—",
         "MRZ Checksum",  "Valid" if fi.get("mrz_checksum_ok") else ("Failed" if fi.get("mrz_checksum_ok") is False else "—")],
        ["SHA-256",       sha256, "", ""],
    ]
    det_table = Table(det_data, colWidths=[3*cm, 5.5*cm, 3*cm, 5.5*cm])
    det_table.setStyle(TableStyle([
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TEXTCOLOR",     (0,0), (0,-1), c_gray),
        ("TEXTCOLOR",     (2,0), (2,-1), c_gray),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica-Bold"),
        ("FONTNAME",      (3,0), (3,-1), "Helvetica-Bold"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LINEBELOW",     (0,0), (-1,-1), 0.25, c_border),
        ("BACKGROUND",    (0,0), (-1,-1), c_light),
    ]))
    story.append(det_table)
    story.append(Spacer(1, 0.3*cm))

    # ── Forensic scores ───────────────────────────────────────────────────────
    story.append(Paragraph("2. FORENSIC ANALYSIS", h2))
    for_data = [
        ["Check", "Result", "Detail"],
        ["ELA Mean Difference", str(ela.get("ela_mean_diff","—")),
         "High = edited regions detected"],
        ["ELA Std Deviation", str(ela.get("ela_std_diff","—")),
         "High = localised tampering"],
        ["Hologram Detection",
         "Detected" if holo.get("holographic_patch_detected") else "NOT DETECTED",
         "Security feature check"],
        ["FFT Peak Ratio", str(round(holo.get("fft_peak_ratio",0),2)),
         "Security background pattern"],
        ["Template Match", str(tmpl.get("template_matched","—")),
         "Country template validation"],
        ["Keyword Ratio", str(tmpl.get("keyword_ratio","—")),
         "Expected text fields found"],
        ["ML Prediction", str(ml.get("ml_prediction","—")),
         "ML tamper score: " + str(ml.get("ml_tamper_score","—"))],
        ["Face Match",
         "VERIFIED" if face.get("face_match") else ("FAILED" if face.get("face_match") is False else "—"),
         "Similarity: " + str(face.get("similarity_pct","—")) + "%"],
    ]
    for_table = Table(for_data, colWidths=[5*cm, 4*cm, 8*cm])
    for_table.setStyle(TableStyle([
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,0), (-1,0),  c_dark),
        ("TEXTCOLOR",     (0,0), (-1,0),  white),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LINEBELOW",     (0,0), (-1,-1), 0.25, c_border),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, c_light]),
    ]))
    story.append(for_table)
    story.append(Spacer(1, 0.3*cm))

    # ── Fraud flags ───────────────────────────────────────────────────────────
    story.append(Paragraph(f"3. FRAUD FLAGS DETECTED ({len(flags)})", h2))
    if flags:
        flag_data = [[Paragraph(f, code)] for f in flags]
        flag_table = Table(flag_data, colWidths=[17*cm])
        flag_table.setStyle(TableStyle([
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [HexColor("#fef2f2"), white]),
            ("LINEBELOW",     (0,0), (-1,-1), 0.25, c_border),
        ]))
        story.append(flag_table)
    else:
        story.append(Paragraph("No fraud flags detected.", body))
    story.append(Spacer(1, 0.3*cm))

    # ── Staff decision ────────────────────────────────────────────────────────
    story.append(Paragraph("4. STAFF DECISION", h2))
    dec_data = [
        ["Decision",   staff_dec.get("decision", "PENDING REVIEW")],
        ["Notes",      staff_dec.get("notes", "—")],
        ["Decided At", staff_dec.get("decided_at", "—")[:19] if staff_dec.get("decided_at") else "—"],
    ]
    dec_table = Table(dec_data, colWidths=[4*cm, 13*cm])
    dec_table.setStyle(TableStyle([
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TEXTCOLOR",     (0,0), (0,-1), c_gray),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica-Bold"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LINEBELOW",     (0,0), (-1,-1), 0.25, c_border),
        ("BACKGROUND",    (0,0), (-1,-1), c_light),
    ]))
    story.append(dec_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%d %B %Y %H:%M UTC')} | "
        "KYC Fraud Detection System | CONFIDENTIAL — FOR INTERNAL USE ONLY",
        style("footer", fontSize=7, textColor=c_gray, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()
