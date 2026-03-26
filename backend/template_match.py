"""
template_match.py — Country-specific ID card template matching

Strategy (no cloud, no giant model required):
  1. Classify document country/type via OCR keyword + MRZ country code
  2. Compare structural layout against expected zones using ORB feature matching
  3. Validate expected text zone positions (aspect ratio, zone presence)
  4. Check characteristic colour palette for known ID types
  5. Validate MRZ type (TD1/TD2/TD3/MRVA/MRVB) and line count

Templates are defined as lightweight descriptor dicts — no stored images needed.
Add new countries by extending TEMPLATES below.
"""
import re
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field

log = logging.getLogger("fraud_detect.template")


# ── Template definitions ───────────────────────────────────────────────────────

@dataclass
class IDTemplate:
    country_code: str          # ISO 3166-1 alpha-3 (matches MRZ field)
    country_name: str
    doc_types: list[str]       # e.g. ["PASSPORT", "NATIONAL ID"]
    mrz_type: str              # TD1 | TD2 | TD3 | MRVA | MRVB
    mrz_lines: int             # expected number of MRZ lines
    mrz_line_len: int          # expected chars per MRZ line
    # aspect ratio of the full document (width/height), ±15% tolerance
    aspect_ratio: float
    # expected primary background colour in HSV (h_mean, s_mean, v_mean) ±20 tolerance
    bg_hsv: tuple[float, float, float] | None = None
    # keywords expected to appear in OCR text
    keywords: list[str] = field(default_factory=list)
    # zone layout: list of (name, x_frac, y_frac, w_frac, h_frac) — fractions of image
    zones: list[tuple] = field(default_factory=list)


TEMPLATES: dict[str, IDTemplate] = {
    # ── Passports (TD3: 2 lines × 44 chars) ──────────────────────────────────
    "PASSPORT_GENERIC": IDTemplate(
        country_code = "ANY",
        country_name = "Generic Passport",
        doc_types    = ["PASSPORT", "P"],
        mrz_type     = "TD3",
        mrz_lines    = 2,
        mrz_line_len = 44,
        aspect_ratio = 1.42,   # ICAO booklet page
        keywords     = ["PASSPORT", "SURNAME", "GIVEN NAMES", "NATIONALITY",
                        "DATE OF BIRTH", "DATE OF EXPIRY"],
        zones        = [
            ("photo",    0.00, 0.30, 0.30, 0.45),
            ("mrz",      0.00, 0.75, 1.00, 0.25),
            ("data",     0.30, 0.30, 0.70, 0.45),
        ],
    ),
    # ── TD1 cards (3 lines × 30 chars) ───────────────────────────────────────
    "ID_TD1_GENERIC": IDTemplate(
        country_code = "ANY",
        country_name = "Generic TD1 ID Card",
        doc_types    = ["ID", "NATIONAL ID", "IDENTITY CARD"],
        mrz_type     = "TD1",
        mrz_lines    = 3,
        mrz_line_len = 30,
        aspect_ratio = 1.585,  # CR80 card standard
        keywords     = ["IDENTITY", "SURNAME", "DATE OF BIRTH"],
        zones        = [
            ("photo",    0.00, 0.00, 0.28, 0.65),
            ("mrz",      0.00, 0.65, 1.00, 0.35),
            ("data",     0.28, 0.00, 0.72, 0.65),
        ],
    ),
    # ── Visa sticker (MRVA: 2 lines × 44 chars) ──────────────────────────────
    "VISA_MRVA": IDTemplate(
        country_code = "ANY",
        country_name = "Type-A Visa",
        doc_types    = ["VISA"],
        mrz_type     = "MRVA",
        mrz_lines    = 2,
        mrz_line_len = 44,
        aspect_ratio = 1.34,
        keywords     = ["VISA", "VALID FOR", "ENTRIES"],
        zones = [],
    ),
    # ── Cambodia national ID (example country-specific) ───────────────────────
    "KHM_ID": IDTemplate(
        country_code = "KHM",
        country_name = "Cambodia National ID",
        doc_types    = ["IDENTITY CARD", "ID"],
        mrz_type     = "TD1",
        mrz_lines    = 3,
        mrz_line_len = 30,
        aspect_ratio = 1.585,
        bg_hsv       = (110, 40, 200),   # light blue background
        keywords     = ["CAMBODIA", "KINGDOM", "IDENTITY"],
        zones        = [
            ("photo",    0.00, 0.00, 0.28, 0.62),
            ("mrz",      0.00, 0.62, 1.00, 0.38),
            ("data",     0.28, 0.00, 0.72, 0.62),
        ],
    ),
    # ── Thailand national ID ───────────────────────────────────────────────────
    "THA_ID": IDTemplate(
        country_code = "THA",
        country_name = "Thailand National ID",
        doc_types    = ["IDENTITY CARD", "บัตรประจำตัวประชาชน"],
        mrz_type     = "TD1",
        mrz_lines    = 3,
        mrz_line_len = 30,
        aspect_ratio = 1.585,
        bg_hsv       = (20, 30, 230),    # cream/white background
        keywords     = ["THAILAND", "IDENTITY", "THAI"],
        zones        = [
            ("photo",    0.00, 0.00, 0.30, 0.60),
            ("mrz",      0.00, 0.65, 1.00, 0.35),
        ],
    ),
}

MRZ_COUNTRY_MAP = {
    "KHM": "KHM_ID",
    "THA": "THA_ID",
    "P<":  "PASSPORT_GENERIC",
    "V<":  "VISA_MRVA",
}


# ── MRZ type detection ────────────────────────────────────────────────────────

def _detect_mrz_type(mrz_lines: list[str]) -> dict:
    """Identify MRZ format and extract country code from MRZ."""
    info = {}
    if not mrz_lines:
        return info

    line_lengths = [len(l) for l in mrz_lines]
    line_count   = len(mrz_lines)

    if line_count == 2 and all(l == 44 for l in line_lengths):
        info["mrz_format"] = "TD3"
    elif line_count == 3 and all(l == 30 for l in line_lengths):
        info["mrz_format"] = "TD1"
    elif line_count == 2 and all(l == 44 for l in line_lengths):
        info["mrz_format"] = "MRVA"
    else:
        info["mrz_format"] = "UNKNOWN"

    # extract country from MRZ line 1
    try:
        line1 = mrz_lines[0].replace(" ", "")
        if line1[0] == "P":
            info["mrz_doc_type"] = "PASSPORT"
            info["mrz_country"]  = line1[2:5].replace("<", "")
        elif line1[0] in ("I", "A", "C"):
            info["mrz_doc_type"] = "ID_CARD"
            info["mrz_country"]  = line1[2:5].replace("<", "")
        elif line1[0] == "V":
            info["mrz_doc_type"] = "VISA"
            info["mrz_country"]  = line1[2:5].replace("<", "")
    except (IndexError, Exception):
        pass

    return info


# ── Template selection ────────────────────────────────────────────────────────

def _select_template(ocr_text: str, mrz_info: dict) -> IDTemplate | None:
    text_upper = ocr_text.upper()
    country    = mrz_info.get("mrz_country", "")
    doc_type   = mrz_info.get("mrz_doc_type", "")

    # country-specific first
    if country in MRZ_COUNTRY_MAP:
        key = MRZ_COUNTRY_MAP[country]
        if key in TEMPLATES:
            return TEMPLATES[key]

    # generic by doc type
    if doc_type == "PASSPORT":
        return TEMPLATES["PASSPORT_GENERIC"]
    if doc_type == "VISA":
        return TEMPLATES["VISA_MRVA"]
    if doc_type == "ID_CARD":
        return TEMPLATES["ID_TD1_GENERIC"]

    # fallback: keyword scan
    for name, tmpl in TEMPLATES.items():
        if any(kw in text_upper for kw in tmpl.keywords):
            return tmpl

    return None


# ── Zone presence check ───────────────────────────────────────────────────────

def _check_zones(img: np.ndarray, template: IDTemplate) -> dict:
    """Verify expected zones have content (edges/text present)."""
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    results = {}

    for zone_name, xf, yf, wf, hf in template.zones:
        x1 = int(xf * w);  y1 = int(yf * h)
        x2 = int((xf+wf)*w); y2 = int((yf+hf)*h)
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            results[zone_name] = {"present": False, "edge_density": 0}
            continue
        edges = cv2.Canny(region, 50, 150)
        density = float(edges.sum() / 255) / max(region.size, 1)
        results[zone_name] = {
            "present":      density > 0.005,
            "edge_density": round(density, 5),
        }

    return results


# ── Background colour check ───────────────────────────────────────────────────

def _check_bg_colour(img: np.ndarray, template: IDTemplate) -> dict:
    if template.bg_hsv is None:
        return {}
    hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    mean_hsv   = [float(hsv[:,:,c].mean()) for c in range(3)]
    expected   = list(template.bg_hsv)
    delta      = float(np.linalg.norm(np.array(mean_hsv) - np.array(expected)))
    return {
        "bg_colour_delta":    round(delta, 2),
        "bg_colour_expected": expected,
        "bg_colour_actual":   [round(v,2) for v in mean_hsv],
    }


# ── Aspect ratio check ────────────────────────────────────────────────────────

def _check_aspect(img: np.ndarray, template: IDTemplate) -> dict:
    h, w   = img.shape[:2]
    actual = round(w / max(h, 1), 3)
    delta  = abs(actual - template.aspect_ratio) / template.aspect_ratio
    return {
        "aspect_ratio_actual":   actual,
        "aspect_ratio_expected": template.aspect_ratio,
        "aspect_ratio_delta_pct": round(delta * 100, 2),
    }


# ── Main entry ────────────────────────────────────────────────────────────────

def match_template(
    image_path: str,
    ocr_text: str,
    mrz_lines: list[str],
) -> tuple[dict, list[str]]:
    """
    Full template match.  Returns (info_dict, flags_list).
    """
    flags, info = [], {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return info, ["image_read_failed"]

        mrz_info = _detect_mrz_type(mrz_lines)
        info.update(mrz_info)

        template = _select_template(ocr_text, mrz_info)
        if template is None:
            flags.append("no_matching_template")
            info["template_matched"] = None
            return info, flags

        info["template_matched"] = template.country_name
        info["template_mrz_type"] = template.mrz_type

        # ── MRZ line count validation ─────────────────────────────────────────
        if len(mrz_lines) != template.mrz_lines:
            flags.append("mrz_line_count_mismatch")
            info["mrz_lines_found"]    = len(mrz_lines)
            info["mrz_lines_expected"] = template.mrz_lines

        # ── MRZ line length validation ────────────────────────────────────────
        wrong_len = [l for l in mrz_lines if len(l) != template.mrz_line_len]
        if wrong_len:
            flags.append("mrz_line_length_mismatch")
            info["mrz_wrong_length_lines"] = len(wrong_len)

        # ── Aspect ratio ──────────────────────────────────────────────────────
        aspect_info = _check_aspect(img, template)
        info.update(aspect_info)
        if aspect_info["aspect_ratio_delta_pct"] > 15:
            flags.append("aspect_ratio_mismatch")

        # ── Zone presence ─────────────────────────────────────────────────────
        zone_results = _check_zones(img, template)
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

        # ── Keyword presence ──────────────────────────────────────────────────
        text_upper   = ocr_text.upper()
        found_kw     = [kw for kw in template.keywords if kw in text_upper]
        missing_kw   = [kw for kw in template.keywords if kw not in text_upper]
        info["keywords_found"]   = found_kw
        info["keywords_missing"] = missing_kw
        kw_ratio = len(found_kw) / max(len(template.keywords), 1)
        if kw_ratio < 0.4:
            flags.append("low_keyword_match")

    except Exception as e:
        flags.append("template_match_failed")
        info["template_error"] = str(e)

    return info, flags
