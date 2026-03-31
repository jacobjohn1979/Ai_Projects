"""
screening.py — All document screening functions
Shared module imported by both app.py (FastAPI) and tasks.py (Celery workers).
Keeping functions here avoids circular imports between app and tasks.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

import cv2
import fitz
import numpy as np
import pikepdf
import pytesseract
from PIL import Image, ExifTags, ImageChops
from dateutil import parser as date_parser

log = logging.getLogger("fraud_detect.screening")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ── constants ─────────────────────────────────────────────────────────────────
SUSPICIOUS_PDF_TOOLS = ["photoshop","illustrator","canva","word","powerpoint",
                        "libreoffice","inkscape","gimp","paint","affinity","coreldraw"]
SUSPICIOUS_IMAGE_TOOLS = ["photoshop","illustrator","canva","snapseed","picsart",
                           "pixlr","gimp","lightroom","affinity","facetune","meitu"]

AMOUNT_REGEX = r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b"
DATE_REGEX   = r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"
ID_REGEX     = r"\b[A-Z0-9]{6,20}\b"
MRZ_LINE_RE  = r"^[A-Z0-9<]{30,44}$"


# ── helpers ───────────────────────────────────────────────────────────────────

def _tmp(suffix=".jpg"):
    return UPLOAD_DIR / f"tmp_{uuid.uuid4().hex}{suffix}"

def _remove(p):
    try: Path(p).unlink(missing_ok=True)
    except Exception: pass

def _parse_pdf_date(raw: Any):
    if not raw: return None
    text = re.sub(r"D:", "", str(raw))
    text = re.sub(r"([+-]\d{2})'(\d{2})'?$", r"\1:\2", text)
    try: return date_parser.parse(text)
    except Exception: return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_pdf_forensics(file_path: str):
    flags, info = [], {}
    try:
        with pikepdf.open(file_path) as pdf:
            meta     = pdf.docinfo or {}
            creator  = str(meta.get("/Creator",  "")).strip()
            producer = str(meta.get("/Producer", "")).strip()
            created  = _parse_pdf_date(meta.get("/CreationDate"))
            modified = _parse_pdf_date(meta.get("/ModDate"))

            info.update(creator=creator, producer=producer, encrypted=pdf.is_encrypted,
                        page_count=len(pdf.pages), pdf_version=str(pdf.pdf_version),
                        created=created.isoformat()  if created  else None,
                        modified=modified.isoformat() if modified else None)

            tool_text = f"{creator} {producer}".lower()
            matched = [t for t in SUSPICIOUS_PDF_TOOLS if t in tool_text]
            if matched: flags.append(f"suspicious_tool:{','.join(matched)}")

            if created and modified:
                delta = (modified - created).days
                if delta > 0:   flags.append("modified_after_created")
                if delta > 365: flags.append("modified_long_after_creation")

            if not creator and not producer: flags.append("missing_creator_producer")
            if pdf.is_encrypted:             flags.append("pdf_is_encrypted")

            for page in pdf.pages:
                if "/AA" in page or "/JavaScript" in str(page.get("/AA", "")):
                    flags.append("javascript_in_pdf"); break

    except Exception as e:
        flags.append("pdf_forensic_parse_error")
        info["forensic_error"] = str(e)
    return info, flags


def extract_text_and_layout(file_path: str):
    flags, pages, full_text = [], [], []
    scanned_like_pages = 0
    try:
        doc = fitz.open(file_path)
        all_fonts = []

        for i, page in enumerate(doc):
            text   = page.get_text("text").strip()
            blocks = page.get_text("dict")["blocks"]
            images = page.get_images(full=True)
            area   = max(page.rect.width * page.rect.height, 1)
            annots = list(page.annots())

            for b in blocks:
                if b.get("type") == 0:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            fn = span.get("font", "")
                            if fn: all_fonts.append(fn.lower())

            large_image = any(
                (b.width * b.height) / area > 0.60
                for img in images
                for b in (page.get_image_rects(img[0]) or [])
            )
            is_scanned = len(text) < 30 and large_image
            if is_scanned: scanned_like_pages += 1

            pages.append({"page": i+1, "text_length": len(text), "image_count": len(images),
                          "annotation_count": len(annots),
                          "has_redaction": any(a.type[0] == 12 for a in annots),
                          "scanned_like": is_scanned})
            full_text.append(text)

        unique_fonts = set(all_fonts)
        if len(unique_fonts) > 6: flags.append("excessive_font_diversity")

        combined = "\n".join(full_text).strip()
        if scanned_like_pages == len(doc) > 0: flags.append("image_based_pdf")
        elif scanned_like_pages > 0:           flags.append("mixed_digital_and_scanned_pages")
        if any(p["has_redaction"] for p in pages): flags.append("contains_redaction_annotations")

        return {"pages": pages, "text": combined, "scanned_like_pages": scanned_like_pages,
                "total_pages": len(doc),
                "fonts": {"unique_fonts": list(unique_fonts), "font_count": len(unique_fonts)}
                }, flags
    except Exception as e:
        return {"pages":[], "text":"", "scanned_like_pages":0, "total_pages":0, "fonts":{}
                }, [f"layout_parse_error:{e}"]


def run_ocr_if_needed(file_path: str, scanned_like: bool):
    if not scanned_like: return {"ocr_used": False, "ocr_text": ""}, []
    flags, ocr_text = [], []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            pix  = page.get_pixmap(dpi=250)
            tmp  = _tmp(".png")
            pix.save(str(tmp))
            text = pytesseract.image_to_string(Image.open(tmp), config="--oem 3 --psm 6")
            ocr_text.append(text)
            _remove(tmp)
        return {"ocr_used": True, "ocr_text": "\n".join(ocr_text)}, flags
    except Exception as e:
        return {"ocr_used": True, "ocr_text": "", "ocr_error": str(e)}, ["ocr_failed"]


def field_checks_pdf(text: str):
    flags, findings = [], {}
    amounts = re.findall(AMOUNT_REGEX, text)
    dates   = re.findall(DATE_REGEX, text)
    findings.update(amount_count=len(amounts), date_count=len(dates),
                    amounts=amounts[:10], dates=dates[:10])

    if len(amounts) >= 2:
        vals = [float(a.replace(",","")) for a in amounts]
        if max(vals) / max(min(vals), 0.01) > 1000:
            flags.append("amount_magnitude_inconsistency")

    parsed_dates = []
    for d in dates[:10]:
        try: parsed_dates.append(date_parser.parse(d, dayfirst=True))
        except Exception: pass
    if len(parsed_dates) >= 2:
        if (max(parsed_dates) - min(parsed_dates)).days > 365*5:
            flags.append("date_spread_too_large")

    if not amounts: flags.append("no_amount_pattern_found")
    if not dates:   flags.append("no_date_pattern_found")
    return findings, flags


def score_pdf(flags: list) -> tuple:
    weights = {
        "modified_after_created":20, "modified_long_after_creation":15,
        "missing_creator_producer":5, "image_based_pdf":15,
        "mixed_digital_and_scanned_pages":20, "ocr_failed":10,
        "no_amount_pattern_found":5, "no_date_pattern_found":5,
        "pdf_forensic_parse_error":15, "excessive_font_diversity":20,
        "contains_redaction_annotations":10, "pdf_is_encrypted":10,
        "javascript_in_pdf":35, "amount_magnitude_inconsistency":20,
        "date_spread_too_large":15,
        "duplicate_file_resubmission":30, "id_number_velocity_breach":25,
        "id_previously_flagged_high_risk":30,
    }
    score = sum(weights.get(f.split(":")[0], 8) for f in flags)
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 20 else "LOW")
    return score, level


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE FUNCTIONS
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
                exif_data[ExifTags.TAGS.get(tag_id, str(tag_id))] = str(value)
        info["exif"] = exif_data

        software = exif_data.get("Software", "")
        if software:
            info["software"] = software
            matched = [t for t in SUSPICIOUS_IMAGE_TOOLS if t in software.lower()]
            if matched: flags.append(f"suspicious_editing_software:{','.join(matched)}")
        else:
            flags.append("missing_software_metadata")

        if not exif_data:              flags.append("missing_exif_metadata")
        if "GPSInfo" in exif_data:     flags.append("gps_data_present")
        if img.mode not in ("RGB","L","CMYK"): flags.append("unusual_color_mode")

    except Exception as e:
        flags.append("image_metadata_parse_error")
        info["metadata_error"] = str(e)
    return info, flags


def perform_ela(image_path: str, quality: int = 90):
    flags, info = [], {}
    tmp = _tmp(".jpg")
    try:
        orig = Image.open(image_path).convert("RGB")
        orig.save(str(tmp), "JPEG", quality=quality)
        diff     = ImageChops.difference(orig, Image.open(str(tmp)))
        diff_np  = np.array(diff).astype(np.float32)
        mean_d   = float(diff_np.mean())
        max_d    = int(diff_np.max())
        std_d    = float(diff_np.std())

        info.update(ela_mean_diff=round(mean_d,4), ela_max_diff=max_d, ela_std_diff=round(std_d,4))
        if mean_d > 12: flags.append("high_ela_mean")
        if std_d  > 20: flags.append("high_ela_variance")
        if max_d  > 80: flags.append("extreme_ela_max")
    except Exception as e:
        flags.append("ela_failed"); info["ela_error"] = str(e)
    finally:
        _remove(tmp)
    return info, flags


def noise_analysis(image_path: str):
    flags, info = [], {}
    try:
        img  = cv2.imread(image_path)
        if img is None: return info, ["image_read_failed"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        noise = gray - blur
        h, w  = noise.shape
        quads = [noise[:h//2,:w//2], noise[:h//2,w//2:], noise[h//2:,:w//2], noise[h//2:,w//2:]]
        variances = [float(np.var(q)) for q in quads]
        ratio = max(variances) / max(min(variances), 0.001)
        info.update(noise_quadrant_variances=[round(v,2) for v in variances],
                    noise_variance_ratio=round(ratio,2))
        if ratio > 8: flags.append("inconsistent_noise_pattern")
        elif ratio > 4: flags.append("moderate_noise_inconsistency")
    except Exception as e:
        flags.append("noise_analysis_failed"); info["noise_error"] = str(e)
    return info, flags


def copy_move_detection(image_path: str):
    flags, info = [], {}
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return info, ["image_read_failed"]
        sift = cv2.SIFT_create(nfeatures=500)
        kps, descs = sift.detectAndCompute(img, None)
        if descs is None or len(descs) < 10:
            info["keypoints"] = 0; return info, flags
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descs, descs, k=2)
        suspicious = sum(
            1 for m, n in matches
            if m.queryIdx != m.trainIdx
            and m.distance < 0.75 * n.distance
            and np.hypot(*(np.array(kps[m.queryIdx].pt) - np.array(kps[m.trainIdx].pt))) > 30
        )
        info.update(sift_suspicious_matches=suspicious, total_keypoints=len(kps))
        if suspicious > 20: flags.append("copy_move_detected")
        elif suspicious > 8: flags.append("possible_copy_move")
    except Exception as e:
        flags.append("copy_move_analysis_failed"); info["copy_move_error"] = str(e)
    return info, flags


def basic_image_forensics(image_path: str):
    flags, info = [], {}
    try:
        img  = cv2.imread(image_path)
        if img is None: return info, ["image_read_failed"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        h, w = gray.shape
        info.update(laplacian_variance=round(lap_var,4), resolution=f"{w}x{h}")
        if lap_var < 20: flags.append("very_blurry_or_resaved")
        if h < 300 or w < 300: flags.append("low_resolution_image")
        sample = list(range(8, h-8, 8))[:20]
        bd = sum(float(np.mean(np.abs(gray[r].astype(float)-gray[r-1].astype(float)))) for r in sample)
        bd /= max(len(sample), 1)
        info["dct_boundary_diff"] = round(bd, 4)
        if bd > 15: flags.append("double_jpeg_compression")
    except Exception as e:
        flags.append("basic_forensics_failed"); info["forensics_error"] = str(e)
    return info, flags


def score_image(flags: list) -> tuple:
    weights = {
        "suspicious_editing_software":25, "missing_software_metadata":5,
        "missing_exif_metadata":10, "gps_data_present":8, "unusual_color_mode":5,
        "high_ela_mean":25, "high_ela_variance":20, "extreme_ela_max":15,
        "inconsistent_noise_pattern":30, "moderate_noise_inconsistency":15,
        "copy_move_detected":35, "possible_copy_move":15,
        "very_blurry_or_resaved":10, "low_resolution_image":5,
        "double_jpeg_compression":20, "ocr_failed":10, "ela_failed":10,
        "image_read_failed":20, "no_date_pattern_found":5,
        "no_id_like_pattern_found":5, "very_low_ocr_text":10,
        "duplicate_file_resubmission":30, "id_previously_flagged_high_risk":30,
    }
    score = sum(weights.get(f.split(":")[0], 5) for f in flags)
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 20 else "LOW")
    return score, level


# ═══════════════════════════════════════════════════════════════════════════════
#  ID CARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _mrz_checksum(zone: str) -> bool:
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(zone[:-1]):
        if   ch == "<":     val = 0
        elif ch.isdigit():  val = int(ch)
        elif ch.isalpha():  val = ord(ch.upper()) - 55
        else:               return False
        total += val * weights[i % 3]
    check = int(zone[-1]) if zone[-1].isdigit() else -1
    return total % 10 == check


def extract_id_card_fields(image_path: str):
    flags, info = [], {}
    try:
        img  = cv2.imread(image_path)
        if img is None: return info, ["id_ocr_failed"]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        _, proc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h, w = gray.shape
        mrz_gray = gray[int(h*0.62):, :]
        _, mrz_proc = cv2.threshold(mrz_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cfg  = "--oem 3 --psm 6"
        text     = pytesseract.image_to_string(proc,     lang="eng", config=cfg)
        mrz_text = pytesseract.image_to_string(mrz_proc, lang="eng", config=cfg)
        combined = f"{text}\n{mrz_text}".strip()

        info.update(ocr_text=combined, text_length=len(combined), mrz_raw_text=mrz_text)

        id_match  = re.search(r"\b\d{8,12}\b", combined)
        dob_match = re.search(r"\b\d{2}[/\-]\d{2}[/\-]\d{4}\b", combined)
        exp_match = re.search(r"(?:EXP|EXPIRES?|VALID UNTIL)[:\s]+(\d{2}[/\-]\d{2}[/\-]\d{4})",
                              combined, re.IGNORECASE)

        info.update(id_number=id_match.group(0)  if id_match  else None,
                    dob=dob_match.group(0)        if dob_match else None,
                    expiry_date=exp_match.group(1) if exp_match else None)

        if not id_match:  flags.append("id_number_not_found")
        if not dob_match: flags.append("dob_not_found")

        if exp_match:
            try:
                exp_dt = date_parser.parse(exp_match.group(1), dayfirst=True)
                if exp_dt < datetime.now():
                    flags.append("id_card_expired")
                if exp_dt.year > datetime.now().year + 15:
                    flags.append("suspicious_far_future_expiry")
            except Exception: pass

        lines     = [l.strip() for l in combined.splitlines() if l.strip()]
        mrz_lines = [l for l in lines if re.match(MRZ_LINE_RE, l)]
        info["mrz_lines"] = mrz_lines

        if len(mrz_lines) >= 2:
            try:
                dob_field = mrz_lines[1][13:20]
                ok = _mrz_checksum(dob_field)
                info["mrz_dob_checksum_ok"] = ok
                if not ok: flags.append("mrz_dob_checksum_failed")
            except Exception: flags.append("mrz_checksum_error")
        else:
            flags.append("mrz_not_detected")

        if id_match and mrz_lines and id_match.group(0) not in " ".join(mrz_lines):
            flags.append("id_number_mrz_mismatch")

        if len(combined.strip()) < 20: flags.append("very_low_id_text")

    except Exception as e:
        flags.append("id_ocr_failed"); info["ocr_error"] = str(e)
    return info, flags


def analyze_id_card_regions(image_path: str):
    flags, info = [], {}
    try:
        img  = cv2.imread(image_path)
        if img is None: return info, ["image_read_failed"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        photo_r = gray[:int(h*.45), :int(w*.28)]
        text_r  = gray[:int(h*.55), int(w*.28):]
        mrz_r   = gray[int(h*.60):, :]

        def lv(r): return float(cv2.Laplacian(r, cv2.CV_64F).var()) if r.size else 0.0
        def nv(r):
            b = cv2.GaussianBlur(r.astype(np.float32), (5,5), 0)
            return float(np.var(r.astype(np.float32) - b)) if r.size else 0.0

        pv, tv, mv = lv(photo_r), lv(text_r), lv(mrz_r)
        info.update(photo_sharpness=round(pv,4), text_sharpness=round(tv,4), mrz_sharpness=round(mv,4))

        if abs(tv - mv) > 150: flags.append("text_mrz_sharpness_mismatch")
        if abs(pv - tv) > 200: flags.append("photo_text_sharpness_mismatch")

        photo_bgr = img[:int(h*.45), :int(w*.28)]
        text_bgr  = img[:int(h*.55), int(w*.28):]
        if photo_bgr.size and text_bgr.size:
            pc = [float(np.mean(photo_bgr[:,:,c])) for c in range(3)]
            tc = [float(np.mean(text_bgr[:,:,c]))  for c in range(3)]
            delta = float(np.linalg.norm(np.array(pc) - np.array(tc)))
            info["color_temperature_delta"] = round(delta, 2)
            if delta > 40: flags.append("color_temperature_mismatch")

        pn, tn, mn = nv(photo_r), nv(text_r), nv(mrz_r)
        info.update(photo_noise=round(pn,4), text_noise=round(tn,4), mrz_noise=round(mn,4))
        if max(pn,tn,mn) / max(min(pn,tn,mn), 0.001) > 10:
            flags.append("noise_fingerprint_mismatch")

    except Exception as e:
        flags.append("region_analysis_failed"); info["region_error"] = str(e)
    return info, flags


def score_id_card(flags: list) -> tuple:
    weights = {
        "id_number_not_found":15, "dob_not_found":10, "mrz_not_detected":20,
        "id_ocr_failed":25, "text_mrz_sharpness_mismatch":25,
        "photo_text_sharpness_mismatch":20, "id_number_mrz_mismatch":35,
        "mrz_dob_checksum_failed":30, "mrz_checksum_error":10,
        "very_low_id_text":10, "id_card_expired":20,
        "suspicious_far_future_expiry":15, "color_temperature_mismatch":25,
        "noise_fingerprint_mismatch":25, "region_analysis_failed":10,
        "image_read_failed":20,
        "high_ela_mean":25, "copy_move_detected":35,
        "inconsistent_noise_pattern":30, "suspicious_editing_software":25,
        "double_jpeg_compression":20,
        "multiple_security_features_absent":35, "no_iridescence_detected":20,
        "missing_security_background_pattern":20, "low_micro_text_density":15,
        "holographic_patch_not_detected":10,
        "mrz_line_count_mismatch":25, "mrz_line_length_mismatch":20,
        "aspect_ratio_mismatch":15, "background_colour_mismatch":15,
        "low_keyword_match":20,
        "face_match_failed":40, "low_face_similarity":30,
        "likely_spoofed_face":40, "possible_spoofed_face":20,
        "no_face_detected":15,
        "duplicate_file_resubmission":30, "id_number_velocity_breach":25,
        "id_previously_flagged_high_risk":35,
    }
    score = sum(weights.get(f.split(":")[0], 5) for f in flags)
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 20 else "LOW")
    return score, level
