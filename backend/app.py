"""
Banking Document Fraud Detection Service — Upgraded
Covers: PDF manipulation, image tampering, ID card forensics
For: KYC / customer onboarding / loan processing
"""

import os
import re
import uuid
import hashlib
import logging
from typing import Any
from pathlib import Path
from datetime import datetime

import cv2
import fitz                        # PyMuPDF
import numpy as np
import pikepdf
import pytesseract
from PIL import Image, ExifTags, ImageChops, ImageFilter
from dateutil import parser as date_parser
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ── optional DB import (graceful fallback) ───────────────────────────────────
try:
    from database import save_screening_log
    DB_ENABLED = True
except ImportError:
    DB_ENABLED = False

load_dotenv()

# ── config ───────────────────────────────────────────────────────────────────
ENV        = os.getenv("ENV", "local")
HOST       = os.getenv("HOST", "127.0.0.1")
PORT       = int(os.getenv("PORT", "8001"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("fraud_detect")

# ── constants ─────────────────────────────────────────────────────────────────
SUSPICIOUS_PDF_TOOLS = ["photoshop", "illustrator", "canva", "word",
                        "powerpoint", "libreoffice", "inkscape", "gimp",
                        "paint", "affinity", "coreldraw"]

SUSPICIOUS_IMAGE_TOOLS = ["photoshop", "illustrator", "canva", "snapseed",
                          "picsart", "pixlr", "gimp", "lightroom", "affinity",
                          "facetune", "meitu"]

AMOUNT_REGEX  = r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b"
DATE_REGEX    = r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"
ID_REGEX      = r"\b[A-Z0-9]{6,20}\b"
MRZ_LINE_RE   = r"^[A-Z0-9<]{30,44}$"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── helpers ───────────────────────────────────────────────────────────────────

def _tmp_path(suffix: str = ".jpg") -> Path:
    return UPLOAD_DIR / f"tmp_{uuid.uuid4().hex}{suffix}"


def _remove(path):
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def parse_pdf_date(raw_value: Any):
    if not raw_value:
        return None
    text = re.sub(r"D:", "", str(raw_value))
    text = re.sub(r"([+-]\d{2})'(\d{2})'?$", r"\1:\2", text)
    try:
        return date_parser.parse(text)
    except Exception:
        return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF SCREENING
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_pdf_forensics(file_path: str):
    """Metadata, creator chain, timestamps, encryption, JavaScript."""
    flags, info = [], {}

    try:
        with pikepdf.open(file_path) as pdf:
            meta   = pdf.docinfo or {}
            info["encrypted"]    = pdf.is_encrypted
            info["page_count"]   = len(pdf.pages)
            info["pdf_version"]  = str(pdf.pdf_version)

            creator  = str(meta.get("/Creator",  "")).strip()
            producer = str(meta.get("/Producer", "")).strip()
            created  = parse_pdf_date(meta.get("/CreationDate"))
            modified = parse_pdf_date(meta.get("/ModDate"))

            info.update(creator=creator, producer=producer,
                        created=created.isoformat()  if created  else None,
                        modified=modified.isoformat() if modified else None)

            # suspicious authoring tools
            tool_text = f"{creator} {producer}".lower()
            matched = [t for t in SUSPICIOUS_PDF_TOOLS if t in tool_text]
            if matched:
                flags.append(f"suspicious_tool:{','.join(matched)}")

            # timestamp anomalies
            if created and modified:
                delta_days = (modified - created).days
                if delta_days > 0:
                    flags.append("modified_after_created")
                if delta_days > 365:
                    flags.append("modified_long_after_creation")

            if not creator and not producer:
                flags.append("missing_creator_producer")

            if pdf.is_encrypted:
                flags.append("pdf_is_encrypted")

            # JavaScript detection (common in weaponised docs)
            raw = pikepdf.Object.parse(b"null")
            for page in pdf.pages:
                if "/AA" in page or "/JavaScript" in str(page.get("/AA", "")):
                    flags.append("javascript_in_pdf")
                    break

    except Exception as e:
        flags.append("pdf_forensic_parse_error")
        info["forensic_error"] = str(e)

    return info, flags


def extract_text_and_layout(file_path: str):
    """Per-page text, images, font analysis, annotation detection."""
    flags, pages, full_text = [], [], []
    scanned_like_pages = 0

    try:
        doc = fitz.open(file_path)
        all_fonts = []

        for i, page in enumerate(doc):
            text   = page.get_text("text").strip()
            blocks = page.get_text("dict")["blocks"]
            images = page.get_images(full=True)
            rect   = page.rect
            page_area = max(rect.width * rect.height, 1)

            # collect fonts
            for b in blocks:
                if b.get("type") == 0:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            fname = span.get("font", "")
                            if fname:
                                all_fonts.append(fname.lower())

            # detect annotations (redactions, comments, hidden text)
            annots = list(page.annots())
            has_redaction = any(a.type[0] == 12 for a in annots)  # type 12 = Redact

            large_image = False
            for img in images:
                try:
                    for b in page.get_image_rects(img[0]):
                        if (b.width * b.height) / page_area > 0.60:
                            large_image = True
                except Exception:
                    pass

            is_scanned = len(text) < 30 and large_image
            if is_scanned:
                scanned_like_pages += 1

            pages.append({
                "page": i + 1,
                "text_length": len(text),
                "image_count": len(images),
                "annotation_count": len(annots),
                "has_redaction": has_redaction,
                "scanned_like": is_scanned,
            })
            full_text.append(text)

        # font diversity — many different fonts = potential pasted content
        unique_fonts = set(all_fonts)
        info_fonts = {"unique_fonts": list(unique_fonts), "font_count": len(unique_fonts)}
        if len(unique_fonts) > 6:
            flags.append("excessive_font_diversity")

        combined_text = "\n".join(full_text).strip()

        if scanned_like_pages == len(doc) > 0:
            flags.append("image_based_pdf")
        elif scanned_like_pages > 0:
            flags.append("mixed_digital_and_scanned_pages")

        if any(p["has_redaction"] for p in pages):
            flags.append("contains_redaction_annotations")

        return {
            "pages": pages,
            "text": combined_text,
            "scanned_like_pages": scanned_like_pages,
            "total_pages": len(doc),
            "fonts": info_fonts,
        }, flags

    except Exception as e:
        return {"pages": [], "text": "", "scanned_like_pages": 0,
                "total_pages": 0, "fonts": {}}, [f"layout_parse_error:{e}"]


def run_ocr_if_needed(file_path: str, scanned_like: bool):
    if not scanned_like:
        return {"ocr_used": False, "ocr_text": ""}, []

    flags, ocr_text = [], []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            pix      = page.get_pixmap(dpi=250)
            img_path = _tmp_path(".png")
            pix.save(str(img_path))
            text = pytesseract.image_to_string(Image.open(img_path),
                                               config="--oem 3 --psm 6")
            ocr_text.append(text)
            _remove(img_path)

        return {"ocr_used": True, "ocr_text": "\n".join(ocr_text)}, flags

    except Exception as e:
        return {"ocr_used": True, "ocr_text": "", "ocr_error": str(e)}, ["ocr_failed"]


def field_checks_pdf(text: str):
    """Validate expected financial document fields."""
    flags, findings = [], {}

    amounts = re.findall(AMOUNT_REGEX, text)
    dates   = re.findall(DATE_REGEX, text)

    findings["amount_count"] = len(amounts)
    findings["date_count"]   = len(dates)
    findings["amounts"]      = amounts[:10]
    findings["dates"]        = dates[:10]

    # amount consistency: wildly different magnitudes suggest edits
    if len(amounts) >= 2:
        vals = [float(a.replace(",", "")) for a in amounts]
        if max(vals) / max(min(vals), 0.01) > 1000:
            flags.append("amount_magnitude_inconsistency")

    # date ordering sanity
    parsed_dates = []
    for d in dates[:10]:
        try:
            parsed_dates.append(date_parser.parse(d, dayfirst=True))
        except Exception:
            pass

    if len(parsed_dates) >= 2:
        spread_days = (max(parsed_dates) - min(parsed_dates)).days
        if spread_days > 365 * 5:
            flags.append("date_spread_too_large")

    if not amounts:
        flags.append("no_amount_pattern_found")
    if not dates:
        flags.append("no_date_pattern_found")

    return findings, flags


def score_pdf(flags: list[str]) -> tuple[int, str]:
    weights = {
        "modified_after_created":           20,
        "modified_long_after_creation":     15,
        "missing_creator_producer":          5,
        "image_based_pdf":                  15,
        "mixed_digital_and_scanned_pages":  20,
        "ocr_failed":                       10,
        "no_amount_pattern_found":           5,
        "no_date_pattern_found":             5,
        "pdf_forensic_parse_error":         15,
        "excessive_font_diversity":         20,
        "contains_redaction_annotations":   10,
        "pdf_is_encrypted":                 10,
        "javascript_in_pdf":               35,
        "amount_magnitude_inconsistency":   20,
        "date_spread_too_large":            15,
    }
    score = sum(weights.get(f.split(":")[0], 8) for f in flags)
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 20 else "LOW")
    return score, level


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE SCREENING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_image_metadata(image_path: str):
    flags, info = [], {}
    try:
        img = Image.open(image_path)
        info.update(format=img.format, mode=img.mode, size=img.size)

        exif_data = {}
        raw_exif = img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_data[tag] = str(value)
        info["exif"] = exif_data

        software = exif_data.get("Software", "")
        if software:
            info["software"] = software
            matched = [t for t in SUSPICIOUS_IMAGE_TOOLS if t in software.lower()]
            if matched:
                flags.append(f"suspicious_editing_software:{','.join(matched)}")
        else:
            flags.append("missing_software_metadata")

        if not exif_data:
            flags.append("missing_exif_metadata")

        # GPS stripping (legit bank docs rarely have GPS; forged scans sometimes do)
        if "GPSInfo" in exif_data:
            flags.append("gps_data_present")

        # colour profile check
        if img.mode not in ("RGB", "L", "CMYK"):
            flags.append("unusual_color_mode")

    except Exception as e:
        flags.append("image_metadata_parse_error")
        info["metadata_error"] = str(e)

    return info, flags


def perform_ela(image_path: str, quality: int = 90):
    """Error Level Analysis — detects re-saved/composited regions."""
    flags, info = [], {}
    tmp = _tmp_path(".jpg")
    try:
        original = Image.open(image_path).convert("RGB")
        original.save(str(tmp), "JPEG", quality=quality)
        recompressed = Image.open(str(tmp))
        diff = ImageChops.difference(original, recompressed)

        diff_np = np.array(diff).astype(np.float32)
        mean_diff = float(diff_np.mean())
        max_diff  = int(diff_np.max())
        std_diff  = float(diff_np.std())

        info.update(ela_mean_diff=round(mean_diff, 4),
                    ela_max_diff=max_diff,
                    ela_std_diff=round(std_diff, 4))

        if mean_diff > 12:
            flags.append("high_ela_mean")
        if std_diff > 20:
            flags.append("high_ela_variance")        # localised edits
        if max_diff > 80:
            flags.append("extreme_ela_max")

    except Exception as e:
        flags.append("ela_failed")
        info["ela_error"] = str(e)
    finally:
        _remove(tmp)

    return info, flags


def noise_analysis(image_path: str):
    """Inconsistent noise = composited/cloned regions."""
    flags, info = [], {}
    try:
        img = cv2.imread(image_path)
        if img is None:
            return info, ["image_read_failed"]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # high-pass residual noise
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        noise  = gray - blur

        # split into quadrants and compare noise variance
        h, w = noise.shape
        quads = [
            noise[:h//2, :w//2], noise[:h//2, w//2:],
            noise[h//2:, :w//2], noise[h//2:, w//2:],
        ]
        variances = [float(np.var(q)) for q in quads]
        info["noise_quadrant_variances"] = [round(v, 2) for v in variances]

        max_v, min_v = max(variances), min(variances)
        ratio = max_v / max(min_v, 0.001)
        info["noise_variance_ratio"] = round(ratio, 2)

        if ratio > 8:
            flags.append("inconsistent_noise_pattern")   # strong splice indicator
        elif ratio > 4:
            flags.append("moderate_noise_inconsistency")

    except Exception as e:
        flags.append("noise_analysis_failed")
        info["noise_error"] = str(e)

    return info, flags


def copy_move_detection(image_path: str):
    """Detect copy-move forgery via SIFT keypoint clustering."""
    flags, info = [], {}
    try:
        img  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return info, ["image_read_failed"]

        sift  = cv2.SIFT_create(nfeatures=500)
        kps, descs = sift.detectAndCompute(img, None)

        if descs is None or len(descs) < 10:
            info["keypoints"] = 0
            return info, flags

        bf      = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descs, descs, k=2)

        suspicious = 0
        for m, n in matches:
            if m.queryIdx == m.trainIdx:
                continue
            if m.distance < 0.75 * n.distance:
                pt1 = kps[m.queryIdx].pt
                pt2 = kps[m.trainIdx].pt
                dist = np.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])
                if dist > 30:
                    suspicious += 1

        info["sift_suspicious_matches"] = suspicious
        info["total_keypoints"] = len(kps)

        if suspicious > 20:
            flags.append("copy_move_detected")
        elif suspicious > 8:
            flags.append("possible_copy_move")

    except Exception as e:
        flags.append("copy_move_analysis_failed")
        info["copy_move_error"] = str(e)

    return info, flags


def basic_image_forensics(image_path: str):
    flags, info = [], {}
    try:
        img = cv2.imread(image_path)
        if img is None:
            return info, ["image_read_failed"]

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        h, w    = gray.shape

        info.update(laplacian_variance=round(lap_var, 4),
                    resolution=f"{w}x{h}")

        if lap_var < 20:
            flags.append("very_blurry_or_resaved")
        if h < 300 or w < 300:
            flags.append("low_resolution_image")

        # JPEG double-compression artefacts via DCT
        # check blockiness at 8-pixel boundaries
        block_diff = 0.0
        sample_rows = range(8, h - 8, 8)
        for r in list(sample_rows)[:20]:
            block_diff += float(np.mean(np.abs(
                gray[r, :].astype(float) - gray[r-1, :].astype(float)
            )))
        info["dct_boundary_diff"] = round(block_diff / max(len(list(sample_rows)[:20]), 1), 4)
        if info["dct_boundary_diff"] > 15:
            flags.append("double_jpeg_compression")

    except Exception as e:
        flags.append("basic_forensics_failed")
        info["forensics_error"] = str(e)

    return info, flags


def score_image(flags: list[str]) -> tuple[int, str]:
    weights = {
        "suspicious_editing_software":     25,
        "missing_software_metadata":        5,
        "missing_exif_metadata":           10,
        "gps_data_present":                 8,
        "unusual_color_mode":               5,
        "high_ela_mean":                   25,
        "high_ela_variance":               20,
        "extreme_ela_max":                 15,
        "inconsistent_noise_pattern":      30,
        "moderate_noise_inconsistency":    15,
        "copy_move_detected":              35,
        "possible_copy_move":              15,
        "very_blurry_or_resaved":          10,
        "low_resolution_image":             5,
        "double_jpeg_compression":         20,
        "ocr_failed":                      10,
        "ela_failed":                      10,
        "image_read_failed":               20,
        "no_date_pattern_found":            5,
        "no_id_like_pattern_found":         5,
        "very_low_ocr_text":               10,
    }
    score = sum(weights.get(f.split(":")[0], 5) for f in flags)
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 20 else "LOW")
    return score, level


# ═══════════════════════════════════════════════════════════════════════════════
#  ID CARD FORENSICS
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_full(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)   # preserves edges better
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _preprocess_mrz(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mrz  = gray[int(h * 0.62):h, :]
    mrz  = cv2.equalizeHist(mrz)
    _, mrz = cv2.threshold(mrz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mrz


def _mrz_checksum(zone: str) -> bool:
    """ICAO 9303 checksum validation for MRZ fields."""
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(zone[:-1]):
        if ch == "<":
            val = 0
        elif ch.isdigit():
            val = int(ch)
        elif ch.isalpha():
            val = ord(ch.upper()) - 55
        else:
            return False
        total += val * weights[i % 3]
    check = int(zone[-1]) if zone[-1].isdigit() else -1
    return total % 10 == check


def extract_id_card_fields(image_path: str):
    flags, info = [], {}
    try:
        proc     = _preprocess_full(image_path)
        mrz_proc = _preprocess_mrz(image_path)

        if proc is None:
            return info, ["id_ocr_failed"]

        cfg  = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(proc, lang="eng", config=cfg)
        mrz_text = pytesseract.image_to_string(mrz_proc, lang="eng", config=cfg) if mrz_proc is not None else ""

        combined = f"{text}\n{mrz_text}".strip()
        info.update(ocr_text=combined, text_length=len(combined), mrz_raw_text=mrz_text)

        # ── field extraction ─────────────────────────────────────────────────
        id_match  = re.search(r"\b\d{8,12}\b", combined)
        dob_match = re.search(r"\b\d{2}[/\-]\d{2}[/\-]\d{4}\b", combined)
        exp_match = re.search(r"\b(?:EXP|EXPIRES?|VALID UNTIL)[:\s]+(\d{2}[/\-]\d{2}[/\-]\d{4})\b",
                              combined, re.IGNORECASE)

        info["id_number"]   = id_match.group(0)  if id_match  else None
        info["dob"]         = dob_match.group(0) if dob_match else None
        info["expiry_date"] = exp_match.group(1) if exp_match else None

        if not id_match:
            flags.append("id_number_not_found")
        if not dob_match:
            flags.append("dob_not_found")

        # expiry sanity
        if exp_match:
            try:
                exp_dt = date_parser.parse(exp_match.group(1), dayfirst=True)
                if exp_dt < datetime.now():
                    flags.append("id_card_expired")
                if exp_dt.year > datetime.now().year + 15:
                    flags.append("suspicious_far_future_expiry")
            except Exception:
                pass

        # ── MRZ parsing ───────────────────────────────────────────────────────
        lines     = [l.strip() for l in combined.splitlines() if l.strip()]
        mrz_lines = [l for l in lines if re.match(MRZ_LINE_RE, l)]
        info["mrz_lines"] = mrz_lines

        checksum_ok = False
        if len(mrz_lines) >= 2:
            # try checksum on DOB field (pos 13-19 of line 2 in TD3)
            try:
                line2 = mrz_lines[1]
                dob_field = line2[13:20]       # YYMMDD + check digit
                checksum_ok = _mrz_checksum(dob_field)
                info["mrz_dob_checksum_ok"] = checksum_ok
                if not checksum_ok:
                    flags.append("mrz_dob_checksum_failed")
            except Exception:
                flags.append("mrz_checksum_error")
        else:
            flags.append("mrz_not_detected")

        # cross-check: ID number in MRZ
        if id_match and mrz_lines:
            if id_match.group(0) not in " ".join(mrz_lines):
                flags.append("id_number_mrz_mismatch")

        if len(combined.strip()) < 20:
            flags.append("very_low_id_text")

    except Exception as e:
        flags.append("id_ocr_failed")
        info["ocr_error"] = str(e)

    return info, flags


def analyze_id_card_regions(image_path: str):
    """Sharpness, colour temperature, and noise consistency across card zones."""
    flags, info = [], {}
    try:
        img  = cv2.imread(image_path)
        if img is None:
            return info, ["image_read_failed"]

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w  = gray.shape

        photo_region = gray[:int(h*0.45), :int(w*0.28)]
        text_region  = gray[:int(h*0.55), int(w*0.28):]
        mrz_region   = gray[int(h*0.60):, :]

        def lap_var(r): return float(cv2.Laplacian(r, cv2.CV_64F).var()) if r.size else 0.0
        def noise_var(r):
            blur  = cv2.GaussianBlur(r.astype(np.float32), (5,5), 0)
            return float(np.var(r.astype(np.float32) - blur)) if r.size else 0.0

        pv = lap_var(photo_region)
        tv = lap_var(text_region)
        mv = lap_var(mrz_region)

        info.update(photo_sharpness=round(pv,4),
                    text_sharpness=round(tv,4),
                    mrz_sharpness=round(mv,4))

        if abs(tv - mv) > 150:
            flags.append("text_mrz_sharpness_mismatch")
        if abs(pv - tv) > 200:
            flags.append("photo_text_sharpness_mismatch")

        # colour temperature consistency across zones
        def avg_color(region_bgr):
            return [float(np.mean(region_bgr[:,:,c])) for c in range(3)]

        photo_bgr = img[:int(h*0.45), :int(w*0.28)]
        text_bgr  = img[:int(h*0.55), int(w*0.28):]
        if photo_bgr.size and text_bgr.size:
            pc = avg_color(photo_bgr)
            tc = avg_color(text_bgr)
            color_delta = float(np.linalg.norm(np.array(pc) - np.array(tc)))
            info["color_temperature_delta"] = round(color_delta, 2)
            if color_delta > 40:
                flags.append("color_temperature_mismatch")    # photo is pasted

        # noise fingerprint comparison
        pn = noise_var(photo_region)
        tn = noise_var(text_region)
        mn = noise_var(mrz_region)
        info.update(photo_noise=round(pn,4), text_noise=round(tn,4), mrz_noise=round(mn,4))
        if max(pn, tn, mn) / max(min(pn, tn, mn), 0.001) > 10:
            flags.append("noise_fingerprint_mismatch")

    except Exception as e:
        flags.append("region_analysis_failed")
        info["region_error"] = str(e)

    return info, flags


def score_id_card(flags: list[str]) -> tuple[int, str]:
    weights = {
        "id_number_not_found":              15,
        "dob_not_found":                    10,
        "mrz_not_detected":                 20,
        "id_ocr_failed":                    25,
        "text_mrz_sharpness_mismatch":      25,
        "photo_text_sharpness_mismatch":    20,
        "id_number_mrz_mismatch":           35,
        "mrz_dob_checksum_failed":          30,
        "mrz_checksum_error":               10,
        "very_low_id_text":                 10,
        "id_card_expired":                  20,
        "suspicious_far_future_expiry":     15,
        "color_temperature_mismatch":       25,
        "noise_fingerprint_mismatch":       25,
        "region_analysis_failed":           10,
        "image_read_failed":                20,
        # image-level flags also included
        "high_ela_mean":                    25,
        "copy_move_detected":               35,
        "inconsistent_noise_pattern":       30,
        "suspicious_editing_software":      25,
        "double_jpeg_compression":          20,
    }
    score = sum(weights.get(f.split(":")[0], 5) for f in flags)
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 20 else "LOW")
    return score, level


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Banking Document Fraud Detection API",
    description="KYC / Onboarding / Loan — PDF, Image & ID Card screening",
    version="2.0.0",
)


def _save(result: dict, filename: str, category: str, doc_type: str):
    if DB_ENABLED:
        try:
            save_screening_log(result, filename, category, doc_type)
        except Exception as e:
            log.warning(f"DB save failed: {e}")


@app.post("/screen-pdf", summary="Screen a PDF for tampering / manipulation")
async def screen_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    file_id   = uuid.uuid4().hex
    save_path = str(UPLOAD_DIR / f"{file_id}_{file.filename}")

    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())

        sha256 = _sha256(save_path)

        forensic_info, forensic_flags = inspect_pdf_forensics(save_path)
        text_info,     text_flags     = extract_text_and_layout(save_path)

        scanned_like = ("image_based_pdf" in text_flags or
                        "mixed_digital_and_scanned_pages" in text_flags)

        ocr_info, ocr_flags = run_ocr_if_needed(save_path, scanned_like)

        final_text = text_info["text"] or ocr_info.get("ocr_text", "")
        field_info, field_flags = field_checks_pdf(final_text)

        all_flags = forensic_flags + text_flags + ocr_flags + field_flags
        risk_score, risk_level = score_pdf(all_flags)

        result = {
            "file_name":   file.filename,
            "sha256":      sha256,
            "screened_at": datetime.utcnow().isoformat(),
            "risk_score":  risk_score,
            "risk_level":  risk_level,
            "flags":       all_flags,
            "forensics":   forensic_info,
            "text_summary": {
                "total_pages":        text_info["total_pages"],
                "scanned_like_pages": text_info["scanned_like_pages"],
                "text_length":        len(final_text),
                "fonts":              text_info.get("fonts", {}),
            },
            "field_checks": field_info,
            "ocr":          {"used": ocr_info.get("ocr_used", False)},
        }

        _save(result, file.filename, "pdf", "pdf")
        return JSONResponse(result)

    finally:
        _remove(save_path)


@app.post("/screen-image", summary="Screen a document image for forgery")
async def screen_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image type: {ext}")

    file_id   = uuid.uuid4().hex
    save_path = str(UPLOAD_DIR / f"{file_id}_{file.filename}")

    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())

        sha256 = _sha256(save_path)

        metadata_info, metadata_flags = extract_image_metadata(save_path)
        ela_info,      ela_flags      = perform_ela(save_path)
        noise_info,    noise_flags    = noise_analysis(save_path)
        cm_info,       cm_flags       = copy_move_detection(save_path)
        forensic_info, forensic_flags = basic_image_forensics(save_path)

        # OCR for text-bearing documents
        ocr_text = ""
        ocr_flags = []
        try:
            ocr_text = pytesseract.image_to_string(Image.open(save_path))
            if len(ocr_text.strip()) < 10:
                ocr_flags.append("very_low_ocr_text")
            if not re.findall(DATE_REGEX, ocr_text):
                ocr_flags.append("no_date_pattern_found")
            if not re.findall(ID_REGEX, ocr_text):
                ocr_flags.append("no_id_like_pattern_found")
        except Exception:
            ocr_flags.append("ocr_failed")

        all_flags = metadata_flags + ela_flags + noise_flags + cm_flags + forensic_flags + ocr_flags
        risk_score, risk_level = score_image(all_flags)

        result = {
            "file_name":      file.filename,
            "sha256":         sha256,
            "screened_at":    datetime.utcnow().isoformat(),
            "risk_score":     risk_score,
            "risk_level":     risk_level,
            "flags":          all_flags,
            "metadata":       metadata_info,
            "ela":            ela_info,
            "noise_analysis": noise_info,
            "copy_move":      cm_info,
            "image_forensics": forensic_info,
            "ocr_summary":    {"text_length": len(ocr_text), "text_preview": ocr_text[:400]},
        }

        _save(result, file.filename, "image", "image")
        return JSONResponse(result)

    finally:
        _remove(save_path)


@app.post("/screen-id-card", summary="Full forensic screening of ID / passport")
async def screen_id_card(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image type: {ext}")

    file_id   = uuid.uuid4().hex
    save_path = str(UPLOAD_DIR / f"{file_id}_{file.filename}")

    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())

        sha256 = _sha256(save_path)

        metadata_info, metadata_flags = extract_image_metadata(save_path)
        ela_info,      ela_flags      = perform_ela(save_path)
        noise_info,    noise_flags    = noise_analysis(save_path)
        cm_info,       cm_flags       = copy_move_detection(save_path)
        field_info,    field_flags    = extract_id_card_fields(save_path)
        region_info,   region_flags   = analyze_id_card_regions(save_path)

        all_flags = (metadata_flags + ela_flags + noise_flags + cm_flags +
                     field_flags + region_flags)
        risk_score, risk_level = score_id_card(all_flags)

        result = {
            "file_name":    file.filename,
            "sha256":       sha256,
            "screened_at":  datetime.utcnow().isoformat(),
            "risk_score":   risk_score,
            "risk_level":   risk_level,
            "flags":        all_flags,
            "metadata":     metadata_info,
            "ela":          ela_info,
            "noise_analysis": noise_info,
            "copy_move":    cm_info,
            "field_info": {
                "id_number":       field_info.get("id_number"),
                "dob":             field_info.get("dob"),
                "expiry_date":     field_info.get("expiry_date"),
                "mrz_lines":       field_info.get("mrz_lines", []),
                "mrz_checksum_ok": field_info.get("mrz_dob_checksum_ok"),
                "text_length":     field_info.get("text_length", 0),
            },
            "region_info": region_info,
        }

        _save(result, file.filename, "image", "id_card")
        return JSONResponse(result)

    finally:
        _remove(save_path)


@app.get("/", summary="Health check")
def home():
    return {
        "service": "Banking Document Fraud Detection API v2.0",
        "status":  "running",
        "endpoints": ["/screen-pdf", "/screen-image", "/screen-id-card"],
    }
