"""
template_match.py — Country-specific ID card template matching
Supports: Cambodia National ID (old + new), Cambodia Passport, Generic TD1/TD3
"""
import re
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field

log = logging.getLogger("fraud_detect.template")


# ── Template definition ────────────────────────────────────────────────────────

@dataclass
class IDTemplate:
    country_code:  str
    country_name:  str
    doc_types:     list
    mrz_type:      str        # TD1 | TD2 | TD3 | MRVA | MRVB | NONE
    mrz_lines:     int
    mrz_line_len:  int
    aspect_ratio:  float      # width/height ± 15% tolerance
    bg_hsv:        tuple = None   # expected background HSV (h,s,v) ± 25
    keywords:      list = field(default_factory=list)
    khmer_keywords: list = field(default_factory=list)  # Khmer script keywords
    zones:         list = field(default_factory=list)   # (name, xf, yf, wf, hf)
    version:       str = "generic"   # old | new | generic


TEMPLATES = {

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMBODIA — National ID Card (New version, post-2020, biometric)
    # ══════════════════════════════════════════════════════════════════════════
    "KHM_ID_NEW": IDTemplate(
        country_code  = "KHM",
        country_name  = "Cambodia National ID Card (New)",
        doc_types     = ["IDENTITY CARD", "ID", "NATIONAL ID"],
        mrz_type      = "TD1",
        mrz_lines     = 3,
        mrz_line_len  = 30,
        aspect_ratio  = 1.585,        # CR80 standard card
        bg_hsv        = (105, 45, 210),  # light blue background
        keywords      = [
            "KINGDOM OF CAMBODIA", "IDENTITY CARD", "CARTE D'IDENTITE",
            "SURNAME", "GIVEN NAME", "NATIONALITY", "CAMBODIAN",
            "DATE OF BIRTH", "SEX", "PLACE OF BIRTH", "DATE OF EXPIRY",
            "PERSONAL NO", "NATIONAL ID NO",
        ],
        khmer_keywords = [
            "ព្រះរាជាណាចក្រកម្ពុជា",   # Kingdom of Cambodia
            "អត្តសញ្ញាណប័ណ្ណ",         # Identity Card
            "នាម",                      # Name
            "ថ្ងៃខែឆ្នាំកំណើត",         # Date of Birth
            "ភេទ",                      # Sex
            "សញ្ជាតិ",                  # Nationality
            "កម្ពុជា",                  # Cambodia
        ],
        zones = [
            ("photo",      0.00, 0.08, 0.28, 0.58),   # photo top-left
            ("chip",       0.28, 0.08, 0.15, 0.30),   # chip area
            ("name_data",  0.28, 0.08, 0.72, 0.60),   # name and data fields
            ("mrz",        0.00, 0.68, 1.00, 0.32),   # MRZ bottom strip
            ("flag",       0.00, 0.00, 0.12, 0.08),   # Cambodian flag top-left
        ],
        version = "new",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMBODIA — National ID Card (Old version, pre-2020, non-biometric)
    # ══════════════════════════════════════════════════════════════════════════
    "KHM_ID_OLD": IDTemplate(
        country_code  = "KHM",
        country_name  = "Cambodia National ID Card (Old)",
        doc_types     = ["IDENTITY CARD", "ID"],
        mrz_type      = "NONE",       # old cards have no MRZ
        mrz_lines     = 0,
        mrz_line_len  = 0,
        aspect_ratio  = 1.585,
        bg_hsv        = (25, 30, 235),   # cream/white background
        keywords      = [
            "KINGDOM OF CAMBODIA", "IDENTITY CARD",
            "SURNAME", "GIVEN NAME", "DATE OF BIRTH",
            "SEX", "HEIGHT", "BLOOD TYPE", "ADDRESS",
            "NATIONAL IDENTIFICATION NUMBER",
        ],
        khmer_keywords = [
            "ព្រះរាជាណាចក្រកម្ពុជា",
            "អត្តសញ្ញាណប័ណ្ណ",
            "នាម",
            "ថ្ងៃខែឆ្នាំកំណើត",
            "កម្ពុជា",
        ],
        zones = [
            ("photo",      0.00, 0.10, 0.30, 0.55),
            ("name_data",  0.30, 0.10, 0.70, 0.55),
            ("address",    0.00, 0.65, 1.00, 0.25),
            ("signature",  0.60, 0.75, 0.40, 0.20),
        ],
        version = "old",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMBODIA — Passport (TD3: 2 lines × 44 chars)
    # ══════════════════════════════════════════════════════════════════════════
    "KHM_PASSPORT": IDTemplate(
        country_code  = "KHM",
        country_name  = "Cambodia Passport",
        doc_types     = ["PASSPORT"],
        mrz_type      = "TD3",
        mrz_lines     = 2,
        mrz_line_len  = 44,
        aspect_ratio  = 1.42,         # ICAO passport page
        bg_hsv        = (130, 60, 180),  # dark blue cover / light blue data page
        keywords      = [
            "KINGDOM OF CAMBODIA", "PASSPORT", "PASSEPORT",
            "SURNAME", "GIVEN NAMES", "NATIONALITY", "CAMBODIAN",
            "DATE OF BIRTH", "SEX", "PLACE OF BIRTH",
            "DATE OF ISSUE", "DATE OF EXPIRY", "AUTHORITY",
            "PERSONAL NO",
        ],
        khmer_keywords = [
            "ព្រះរាជាណាចក្រកម្ពុជា",
            "លិខិតឆ្លងដែន",    # Passport
            "នាម",
            "ថ្ងៃខែឆ្នាំកំណើត",
            "សញ្ជាតិ",
            "កម្ពុជា",
        ],
        zones = [
            ("photo",      0.00, 0.20, 0.30, 0.50),
            ("data",       0.30, 0.20, 0.70, 0.50),
            ("mrz",        0.00, 0.72, 1.00, 0.28),
            ("header",     0.00, 0.00, 1.00, 0.20),
        ],
        version = "generic",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  GENERIC — Passport TD3 (fallback)
    # ══════════════════════════════════════════════════════════════════════════
    "PASSPORT_GENERIC": IDTemplate(
        country_code  = "ANY",
        country_name  = "Generic Passport (TD3)",
        doc_types     = ["PASSPORT", "P"],
        mrz_type      = "TD3",
        mrz_lines     = 2,
        mrz_line_len  = 44,
        aspect_ratio  = 1.42,
        keywords      = [
            "PASSPORT", "SURNAME", "GIVEN NAMES",
            "NATIONALITY", "DATE OF BIRTH", "DATE OF EXPIRY",
        ],
        zones = [
            ("photo",  0.00, 0.25, 0.30, 0.45),
            ("mrz",    0.00, 0.75, 1.00, 0.25),
            ("data",   0.30, 0.25, 0.70, 0.45),
        ],
        version = "generic",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  GENERIC — TD1 ID Card (fallback)
    # ══════════════════════════════════════════════════════════════════════════
    "ID_TD1_GENERIC": IDTemplate(
        country_code  = "ANY",
        country_name  = "Generic TD1 ID Card",
        doc_types     = ["ID", "IDENTITY CARD", "NATIONAL ID"],
        mrz_type      = "TD1",
        mrz_lines     = 3,
        mrz_line_len  = 30,
        aspect_ratio  = 1.585,
        keywords      = ["IDENTITY", "SURNAME", "DATE OF BIRTH"],
        zones = [
            ("photo",  0.00, 0.00, 0.28, 0.65),
            ("mrz",    0.00, 0.65, 1.00, 0.35),
            ("data",   0.28, 0.00, 0.72, 0.65),
        ],
        version = "generic",
    ),
}

# MRZ country code → template key
MRZ_COUNTRY_MAP = {
    "KHM": ["KHM_ID_NEW", "KHM_ID_OLD", "KHM_PASSPORT"],
}


# ── MRZ type detection ─────────────────────────────────────────────────────────

def _detect_mrz_type(mrz_lines: list) -> dict:
    info = {}
    if not mrz_lines:
        return info

    lengths    = [len(l) for l in mrz_lines]
    line_count = len(mrz_lines)

    if line_count == 2 and all(l == 44 for l in lengths):
        info["mrz_format"] = "TD3"
    elif line_count == 3 and all(l == 30 for l in lengths):
        info["mrz_format"] = "TD1"
    elif line_count == 2 and all(38 <= l <= 44 for l in lengths):
        info["mrz_format"] = "MRVA"
    else:
        info["mrz_format"] = "UNKNOWN"

    try:
        line1 = mrz_lines[0].replace(" ", "")
        if line1[0] == "P":
            info["mrz_doc_type"] = "PASSPORT"
            info["mrz_country"]  = line1[2:5].replace("<", "")
        elif line1[0] in ("I", "A", "C", "D"):
            info["mrz_doc_type"] = "ID_CARD"
            info["mrz_country"]  = line1[2:5].replace("<", "")
        elif line1[0] == "V":
            info["mrz_doc_type"] = "VISA"
            info["mrz_country"]  = line1[2:5].replace("<", "")
    except Exception:
        pass

    return info


# ── Khmer text detection ───────────────────────────────────────────────────────

def _detect_khmer(ocr_text: str) -> dict:
    """Detect presence of Khmer Unicode script in OCR output."""
    # Khmer Unicode range: U+1780–U+17FF
    khmer_chars = [c for c in ocr_text if '\u1780' <= c <= '\u17ff']
    khmer_count = len(khmer_chars)
    return {
        "khmer_chars_found": khmer_count,
        "khmer_detected":    khmer_count > 5,
    }


# ── Template selection ─────────────────────────────────────────────────────────

def _select_template(img: np.ndarray, ocr_text: str, mrz_info: dict) -> IDTemplate:
    """
    Select best matching template using:
    1. MRZ country code
    2. OCR keyword matching
    3. Aspect ratio
    4. Khmer text presence
    """
    text_upper = ocr_text.upper()
    country    = mrz_info.get("mrz_country", "")
    doc_type   = mrz_info.get("mrz_doc_type", "")
    h, w       = img.shape[:2]
    aspect     = w / max(h, 1)

    # ── KHM-specific selection ────────────────────────────────────────────────
    if country == "KHM" or any(kw in text_upper for kw in ["KINGDOM OF CAMBODIA", "CAMBODIAN"]):
        if doc_type == "PASSPORT" or "PASSPORT" in text_upper or "PASSEPORT" in text_upper:
            return TEMPLATES["KHM_PASSPORT"]
        # distinguish new vs old KHM ID by MRZ presence
        if len(mrz_info.get("mrz_format", "")) > 0 and mrz_info.get("mrz_format") != "UNKNOWN":
            return TEMPLATES["KHM_ID_NEW"]
        return TEMPLATES["KHM_ID_OLD"]

    # ── Generic fallback by doc type ─────────────────────────────────────────
    if doc_type == "PASSPORT" or "PASSPORT" in text_upper:
        return TEMPLATES["PASSPORT_GENERIC"]
    if doc_type == "ID_CARD" or any(kw in text_upper for kw in ["IDENTITY CARD", "NATIONAL ID"]):
        return TEMPLATES["ID_TD1_GENERIC"]

    # ── Fallback by aspect ratio ─────────────────────────────────────────────
    if abs(aspect - 1.585) < 0.2:
        return TEMPLATES["ID_TD1_GENERIC"]
    if abs(aspect - 1.42) < 0.2:
        return TEMPLATES["PASSPORT_GENERIC"]

    return TEMPLATES["ID_TD1_GENERIC"]


# ── Zone presence check ────────────────────────────────────────────────────────

def _check_zones(img: np.ndarray, template: IDTemplate) -> dict:
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    results = {}

    for zone_name, xf, yf, wf, hf in template.zones:
        x1 = int(xf * w);       y1 = int(yf * h)
        x2 = int((xf+wf) * w);  y2 = int((yf+hf) * h)
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            results[zone_name] = {"present": False, "edge_density": 0}
            continue
        edges   = cv2.Canny(region, 50, 150)
        density = float(edges.sum() / 255) / max(region.size, 1)
        results[zone_name] = {
            "present":      density > 0.003,
            "edge_density": round(density, 5),
        }

    return results


# ── Background colour check ────────────────────────────────────────────────────

def _check_bg_colour(img: np.ndarray, template: IDTemplate) -> dict:
    if template.bg_hsv is None:
        return {}
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    mean_hsv = [float(hsv[:,:,c].mean()) for c in range(3)]
    expected = list(template.bg_hsv)
    delta    = float(np.linalg.norm(np.array(mean_hsv) - np.array(expected)))
    return {
        "bg_colour_delta":    round(delta, 2),
        "bg_colour_expected": expected,
        "bg_colour_actual":   [round(v, 2) for v in mean_hsv],
    }


# ── Aspect ratio check ─────────────────────────────────────────────────────────

def _check_aspect(img: np.ndarray, template: IDTemplate) -> dict:
    h, w   = img.shape[:2]
    actual = round(w / max(h, 1), 3)
    delta  = abs(actual - template.aspect_ratio) / template.aspect_ratio
    return {
        "aspect_ratio_actual":    actual,
        "aspect_ratio_expected":  template.aspect_ratio,
        "aspect_ratio_delta_pct": round(delta * 100, 2),
    }


# ── Keyword check ──────────────────────────────────────────────────────────────

def _check_keywords(ocr_text: str, template: IDTemplate) -> dict:
    text_upper = ocr_text.upper()
    found      = [kw for kw in template.keywords if kw in text_upper]
    missing    = [kw for kw in template.keywords if kw not in text_upper]
    ratio      = len(found) / max(len(template.keywords), 1)

    # Khmer keyword check
    khmer_found   = [kw for kw in template.khmer_keywords if kw in ocr_text]
    khmer_missing = [kw for kw in template.khmer_keywords if kw not in ocr_text]

    return {
        "keywords_found":    found,
        "keywords_missing":  missing,
        "keyword_ratio":     round(ratio, 2),
        "khmer_found":       khmer_found,
        "khmer_missing":     khmer_missing,
    }


# ── MRZ line validation ────────────────────────────────────────────────────────

def _check_mrz_format(mrz_lines: list, template: IDTemplate) -> tuple:
    flags = []
    info  = {}

    if template.mrz_type == "NONE":
        if mrz_lines:
            flags.append("unexpected_mrz_on_old_id")
        return info, flags

    if len(mrz_lines) != template.mrz_lines:
        flags.append("mrz_line_count_mismatch")
        info["mrz_lines_found"]    = len(mrz_lines)
        info["mrz_lines_expected"] = template.mrz_lines

    wrong_len = [l for l in mrz_lines if len(l) != template.mrz_line_len]
    if wrong_len:
        flags.append("mrz_line_length_mismatch")
        info["mrz_wrong_length_count"] = len(wrong_len)

    return info, flags


# ── Main entry ─────────────────────────────────────────────────────────────────

def match_template(
    image_path: str,
    ocr_text:   str,
    mrz_lines:  list,
) -> tuple:
    """
    Full template match. Returns (info_dict, flags_list).
    """
    flags, info = [], {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return info, ["image_read_failed"]

        # ── MRZ detection ─────────────────────────────────────────────────────
        mrz_info = _detect_mrz_type(mrz_lines)
        info.update(mrz_info)

        # ── Template selection ────────────────────────────────────────────────
        template = _select_template(img, ocr_text, mrz_info)
        info["template_matched"]  = template.country_name
        info["template_version"]  = template.version
        info["template_mrz_type"] = template.mrz_type

        # ── Khmer script detection ────────────────────────────────────────────
        khmer_info = _detect_khmer(ocr_text)
        info.update(khmer_info)

        # ── Aspect ratio ──────────────────────────────────────────────────────
        aspect_info = _check_aspect(img, template)
        info.update(aspect_info)
        if aspect_info["aspect_ratio_delta_pct"] > 15:
            flags.append("aspect_ratio_mismatch")

        # ── Zone presence ─────────────────────────────────────────────────────
        zone_results  = _check_zones(img, template)
        info["zone_checks"] = zone_results
        missing_zones = [z for z, r in zone_results.items() if not r.get("present")]
        if missing_zones:
            flags.append(f"missing_zones:{','.join(missing_zones)}")

        # ── Background colour ─────────────────────────────────────────────────
        bg_info = _check_bg_colour(img, template)
        if bg_info:
            info.update(bg_info)
            if bg_info.get("bg_colour_delta", 0) > 40:
                flags.append("background_colour_mismatch")

        # ── Keywords ──────────────────────────────────────────────────────────
        kw_info = _check_keywords(ocr_text, template)
        info.update(kw_info)
        if kw_info["keyword_ratio"] < 0.35:
            flags.append("low_keyword_match")

        # ── KHM specific: expect some Khmer text ──────────────────────────────
        if template.country_code == "KHM" and not khmer_info["khmer_detected"]:
            flags.append("khmer_text_not_detected")

        # ── MRZ format validation ─────────────────────────────────────────────
        mrz_fmt_info, mrz_fmt_flags = _check_mrz_format(mrz_lines, template)
        info.update(mrz_fmt_info)
        flags.extend(mrz_fmt_flags)

        # ── Old KHM ID specific: should NOT have MRZ ─────────────────────────
        if template.version == "old" and len(mrz_lines) > 0:
            flags.append("mrz_found_on_old_format_id")

        # ── New KHM ID specific: should have chip zone content ────────────────
        if template.version == "new":
            chip_zone = zone_results.get("chip", {})
            if not chip_zone.get("present"):
                flags.append("chip_zone_empty")

    except Exception as e:
        flags.append("template_match_failed")
        info["template_error"] = str(e)
        log.error(f"template_match failed: {e}")

    return info, flags
