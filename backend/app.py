from database import save_screening_log
import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "local")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8001"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


import re
import uuid
from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
import pikepdf
import pytesseract
from PIL import Image, ExifTags, ImageChops
from dateutil import parser as date_parser
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Document Tamper Screening Service")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUSPICIOUS_TOOLS = [
    "photoshop",
    "illustrator",
    "canva",
    "word",
    "powerpoint",
    "libreoffice",
    "inkscape",
]

SUSPICIOUS_IMAGE_TOOLS = [
    "photoshop",
    "illustrator",
    "canva",
    "snapseed",
    "picsart",
]

AMOUNT_REGEX = r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b"
DATE_REGEX = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
ID_REGEX = r"\b[A-Z0-9]{6,20}\b"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def parse_pdf_dates(raw_value: Any):
    if not raw_value:
        return None
    text = str(raw_value)
    text = text.replace("D:", "")
    text = re.sub(r"([+-]\d{2})'(\d{2})'?$", r"\1:\2", text)
    try:
        return date_parser.parse(text)
    except Exception:
        return None


# =========================
# PDF SCREENING FUNCTIONS
# =========================

def inspect_pdf_forensics(file_path: str):
    flags = []
    info = {}

    try:
        with pikepdf.open(file_path) as pdf:
            meta = pdf.docinfo or {}

            creator = str(meta.get("/Creator", "")).strip()
            producer = str(meta.get("/Producer", "")).strip()
            created = parse_pdf_dates(meta.get("/CreationDate"))
            modified = parse_pdf_dates(meta.get("/ModDate"))

            info["creator"] = creator
            info["producer"] = producer
            info["created"] = created.isoformat() if created else None
            info["modified"] = modified.isoformat() if modified else None
            info["page_count"] = len(pdf.pages)

            tool_text = f"{creator} {producer}".lower()
            if any(tool in tool_text for tool in SUSPICIOUS_TOOLS):
                flags.append("suspicious_producer_or_creator")

            if created and modified and modified > created:
                flags.append("modified_after_created")

            if not creator and not producer:
                flags.append("missing_creator_producer")

    except Exception as e:
        flags.append("pdf_forensic_parse_error")
        info["forensic_error"] = str(e)

    return info, flags


def extract_text_and_layout(file_path: str):
    flags = []
    pages = []
    full_text = []
    scanned_like_pages = 0

    doc = fitz.open(file_path)

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        rect = page.rect
        page_area = rect.width * rect.height

        large_image_found = False
        for img in images:
            try:
                xref = img[0]
                bbox_list = page.get_image_rects(xref)
                for b in bbox_list:
                    if (b.width * b.height) / page_area > 0.6:
                        large_image_found = True
                        break
            except Exception:
                pass

        if len(text) < 30 and large_image_found:
            scanned_like_pages += 1

        pages.append({
            "page": i + 1,
            "text_length": len(text),
            "image_count": len(images),
            "scanned_like": len(text) < 30 and large_image_found,
        })
        full_text.append(text)

    combined_text = "\n".join(full_text).strip()

    if scanned_like_pages == len(doc) and len(doc) > 0:
        flags.append("image_based_pdf")
    elif scanned_like_pages > 0:
        flags.append("mixed_digital_and_scanned_pages")

    return {
        "pages": pages,
        "text": combined_text,
        "scanned_like_pages": scanned_like_pages,
        "total_pages": len(doc),
    }, flags


def run_ocr_if_needed(file_path: str, scanned_like: bool):
    if not scanned_like:
        return {"ocr_used": False, "ocr_text": ""}, []

    flags = []
    ocr_text = []

    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img_path = os.path.join(UPLOAD_DIR, f"page_{i+1}_{uuid.uuid4().hex}.png")
            pix.save(img_path)

            text = pytesseract.image_to_string(Image.open(img_path))
            ocr_text.append(text)

            try:
                os.remove(img_path)
            except Exception:
                pass

        return {"ocr_used": True, "ocr_text": "\n".join(ocr_text)}, flags
    except Exception as e:
        flags.append("ocr_failed")
        return {"ocr_used": True, "ocr_text": "", "ocr_error": str(e)}, flags


def field_checks(text: str):
    flags = []
    findings = {}

    amounts = re.findall(AMOUNT_REGEX, text)
    dates = re.findall(DATE_REGEX, text)

    findings["amount_count"] = len(amounts)
    findings["date_count"] = len(dates)

    if len(amounts) == 0:
        flags.append("no_amount_pattern_found")
    if len(dates) == 0:
        flags.append("no_date_pattern_found")

    return findings, flags


def score_pdf_result(flags: list[str]):
    weights = {
        "modified_after_created": 20,
        "suspicious_producer_or_creator": 25,
        "missing_creator_producer": 5,
        "image_based_pdf": 15,
        "mixed_digital_and_scanned_pages": 20,
        "ocr_failed": 10,
        "no_amount_pattern_found": 5,
        "no_date_pattern_found": 5,
        "pdf_forensic_parse_error": 15,
    }

    score = sum(weights.get(f, 5) for f in flags)

    if score >= 50:
        level = "HIGH"
    elif score >= 20:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level


# =========================
# IMAGE SCREENING FUNCTIONS
# =========================

def extract_image_metadata(image_path: str):
    flags = []
    info = {}

    try:
        img = Image.open(image_path)
        info["format"] = img.format
        info["mode"] = img.mode
        info["size"] = img.size

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
            if any(tool in software.lower() for tool in SUSPICIOUS_IMAGE_TOOLS):
                flags.append("suspicious_editing_software")
        else:
            flags.append("missing_software_metadata")

        if not exif_data:
            flags.append("missing_exif_metadata")

    except Exception as e:
        flags.append("image_metadata_parse_error")
        info["metadata_error"] = str(e)

    return info, flags


def run_ocr_on_image(image_path: str):
    flags = []
    info = {}

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        info["ocr_text"] = text
        info["text_length"] = len(text)

        if len(text.strip()) < 10:
            flags.append("very_low_ocr_text")

        if len(re.findall(DATE_REGEX, text)) == 0:
            flags.append("no_date_pattern_found")

        if len(re.findall(ID_REGEX, text)) == 0:
            flags.append("no_id_like_pattern_found")

    except Exception as e:
        flags.append("ocr_failed")
        info["ocr_error"] = str(e)
        info["ocr_text"] = ""

    return info, flags


def perform_ela(image_path: str, quality: int = 90):
    flags = []
    info = {}

    try:
        original = Image.open(image_path).convert("RGB")
        temp_path = os.path.join(UPLOAD_DIR, f"ela_{uuid.uuid4().hex}.jpg")
        original.save(temp_path, "JPEG", quality=quality)

        recompressed = Image.open(temp_path)
        diff = ImageChops.difference(original, recompressed)

        diff_np = np.array(diff)
        mean_diff = float(diff_np.mean())
        max_diff = int(diff_np.max())

        info["ela_mean_diff"] = round(mean_diff, 4)
        info["ela_max_diff"] = max_diff

        if mean_diff > 12:
            flags.append("high_ela_difference")

        try:
            os.remove(temp_path)
        except Exception:
            pass

    except Exception as e:
        flags.append("ela_failed")
        info["ela_error"] = str(e)

    return info, flags


def basic_image_forensics(image_path: str):
    flags = []
    info = {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            flags.append("image_read_failed")
            return info, flags

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        info["laplacian_variance"] = round(float(lap_var), 4)

        if lap_var < 20:
            flags.append("very_blurry_or_resaved_image")

        h, w = gray.shape
        if h < 300 or w < 300:
            flags.append("low_resolution_image")

    except Exception as e:
        flags.append("basic_forensics_failed")
        info["forensics_error"] = str(e)

    return info, flags


def score_image_flags(flags: list[str]):
    weights = {
        "suspicious_editing_software": 25,
        "missing_software_metadata": 5,
        "missing_exif_metadata": 10,
        "very_low_ocr_text": 10,
        "no_date_pattern_found": 5,
        "no_id_like_pattern_found": 5,
        "high_ela_difference": 25,
        "very_blurry_or_resaved_image": 10,
        "low_resolution_image": 5,
        "ocr_failed": 10,
        "ela_failed": 10,
        "image_metadata_parse_error": 10,
        "basic_forensics_failed": 10,
        "image_read_failed": 20,
    }

    score = sum(weights.get(f, 5) for f in flags)

    if score >= 50:
        level = "HIGH"
    elif score >= 20:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level
# =========================
# ID CARD DETECTION
# =========================

def preprocess_for_ocr(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh


def preprocess_mrz_region(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mrz = gray[int(h * 0.62):h, 0:w]
    mrz = cv2.equalizeHist(mrz)
    mrz = cv2.GaussianBlur(mrz, (3, 3), 0)
    _, mrz = cv2.threshold(mrz, 140, 255, cv2.THRESH_BINARY)

    return mrz


def extract_id_card_fields(image_path: str):
    flags = []
    info = {}

    try:
        processed = preprocess_for_ocr(image_path)
        mrz_processed = preprocess_mrz_region(image_path)

        if processed is None:
            flags.append("id_ocr_failed")
            info["ocr_error"] = "Unable to read image"
            return info, flags

        text = pytesseract.image_to_string(processed, lang="eng", config="--psm 6")
        mrz_text = ""
        if mrz_processed is not None:
            mrz_text = pytesseract.image_to_string(mrz_processed, lang="eng", config="--psm 6")

        combined_text = f"{text}\n{mrz_text}".strip()

        info["ocr_text"] = combined_text
        info["text_length"] = len(combined_text)
        info["mrz_raw_text"] = mrz_text

        # ID number
        id_match = re.search(r"\b\d{8,12}\b", combined_text)

        # DOB: dd/mm/yyyy or dd-mm-yyyy
        dob_match = re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", combined_text)

        info["id_number"] = id_match.group(0) if id_match else None
        info["dob"] = dob_match.group(0) if dob_match else None

        if not id_match:
            flags.append("id_number_not_found")
        if not dob_match:
            flags.append("dob_not_found")

        # MRZ-like lines
        lines = [line.strip() for line in combined_text.splitlines() if line.strip()]
        mrz_lines = [line for line in lines if "<" in line and len(line) > 20]

        info["mrz_lines"] = mrz_lines

        if len(mrz_lines) < 2:
            flags.append("mrz_not_detected")

        if len(combined_text.strip()) < 20:
            flags.append("very_low_id_text")

    except Exception as e:
        flags.append("id_ocr_failed")
        info["ocr_error"] = str(e)

    return info, flags


def analyze_id_card_regions(image_path: str):
    flags = []
    info = {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            flags.append("image_read_failed")
            return info, flags

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        photo_region = gray[0:int(h * 0.45), 0:int(w * 0.28)]
        text_region = gray[0:int(h * 0.55), int(w * 0.28):w]
        mrz_region = gray[int(h * 0.60):h, 0:w]

        def lap_var(region):
            return float(cv2.Laplacian(region, cv2.CV_64F).var())

        photo_var = lap_var(photo_region)
        text_var = lap_var(text_region)
        mrz_var = lap_var(mrz_region)

        info["photo_sharpness"] = round(photo_var, 4)
        info["text_sharpness"] = round(text_var, 4)
        info["mrz_sharpness"] = round(mrz_var, 4)

        if abs(text_var - mrz_var) > 150:
            flags.append("text_mrz_sharpness_mismatch")

        if abs(photo_var - text_var) > 200:
            flags.append("photo_text_sharpness_mismatch")

    except Exception as e:
        flags.append("region_analysis_failed")
        info["region_error"] = str(e)

    return info, flags


def validate_id_card_consistency(field_info: dict):
    flags = []
    info = {}

    ocr_text = field_info.get("ocr_text", "")
    id_number = field_info.get("id_number")
    mrz_lines = field_info.get("mrz_lines", [])

    info["mrz_count"] = len(mrz_lines)

    if id_number and mrz_lines:
        mrz_joined = " ".join(mrz_lines)
        if id_number not in mrz_joined:
            flags.append("id_number_mrz_mismatch")

    if len(ocr_text.strip()) < 30:
        flags.append("very_low_id_text")

    # stronger combined rule
    if "very_low_id_text" in flags and len(mrz_lines) == 0:
        flags.append("strong_id_ocr_failure")

    return info, flags


def score_id_card_flags(flags: list[str]):
    weights = {
        "id_number_not_found": 15,
        "dob_not_found": 10,
        "mrz_not_detected": 20,
        "id_ocr_failed": 25,
        "text_mrz_sharpness_mismatch": 25,
        "photo_text_sharpness_mismatch": 20,
        "id_number_mrz_mismatch": 30,
        "very_low_id_text": 10,
        "strong_id_ocr_failure": 20,
        "region_analysis_failed": 10,
        "image_read_failed": 20,
    }

    score = sum(weights.get(f, 5) for f in flags)

    if score >= 50:
        level = "HIGH"
    elif score >= 20:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level

# =========================
# ENDPOINTS
# =========================

@app.post("/screen-pdf")
async def screen_pdf(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    forensic_info, forensic_flags = inspect_pdf_forensics(save_path)
    text_info, text_flags = extract_text_and_layout(save_path)

    scanned_like = (
        "image_based_pdf" in text_flags or
        "mixed_digital_and_scanned_pages" in text_flags
    )

    ocr_info, ocr_flags = run_ocr_if_needed(save_path, scanned_like)

    final_text = text_info["text"]
    if not final_text and ocr_info.get("ocr_text"):
        final_text = ocr_info["ocr_text"]

    field_info, field_flags = field_checks(final_text)

    all_flags = forensic_flags + text_flags + ocr_flags + field_flags
    risk_score, risk_level = score_pdf_result(all_flags)

    # ✅ Build result FIRST
    result = {
        "file_name": file.filename,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flags": all_flags,
        "forensics": forensic_info,
        "text_summary": {
            "total_pages": text_info["total_pages"],
            "scanned_like_pages": text_info["scanned_like_pages"],
            "text_length": len(final_text),
        },
        "field_checks": field_info,
        "ocr": {
            "used": ocr_info.get("ocr_used", False),
        },
    }

    # ✅ Save to DB
    save_screening_log(result, file.filename, "pdf", "pdf")

    # ✅ Return response
    return JSONResponse(result)

from fastapi import File, UploadFile

@app.post("/screen-image")
async def screen_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported image type"}
        )

    file_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    metadata_info, metadata_flags = extract_image_metadata(save_path)
    ocr_info, ocr_flags = run_ocr_on_image(save_path)
    ela_info, ela_flags = perform_ela(save_path)
    forensic_info, forensic_flags = basic_image_forensics(save_path)

    all_flags = metadata_flags + ocr_flags + ela_flags + forensic_flags
    risk_score, risk_level = score_image_flags(all_flags)

    # ✅ Build result FIRST
    result = {
        "file_name": file.filename,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flags": all_flags,
        "metadata": metadata_info,
        "ocr_summary": {
            "text_length": ocr_info.get("text_length", 0),
            "text_preview": ocr_info.get("ocr_text", "")[:300],
        },
        "ela": ela_info,
        "image_forensics": forensic_info,
    }

    # ✅ Save to DB
    save_screening_log(result, file.filename, "image", "image")

    # ✅ Return response
    return JSONResponse(result)
#End Point ID card detection
@app.post("/screen-id-card")
async def screen_id_card(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported image type"}
        )

    file_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    metadata_info, metadata_flags = extract_image_metadata(save_path)
    ela_info, ela_flags = perform_ela(save_path)
    field_info, field_flags = extract_id_card_fields(save_path)
    region_info, region_flags = analyze_id_card_regions(save_path)
    consistency_info, consistency_flags = validate_id_card_consistency(field_info)

    all_flags = (
        metadata_flags +
        ela_flags +
        field_flags +
        region_flags +
        consistency_flags
    )

    risk_score, risk_level = score_id_card_flags(all_flags)

    result = {
        "file_name": file.filename,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flags": all_flags,
        "metadata": metadata_info,
        "ela": ela_info,
        "field_info": {
            "id_number": field_info.get("id_number"),
            "dob": field_info.get("dob"),
            "mrz_lines": field_info.get("mrz_lines", []),
            "text_length": field_info.get("text_length", 0),
        },
        "region_info": region_info,
        "consistency_info": consistency_info,
    }

    save_screening_log(result, file.filename, "image", "id_card")
    return JSONResponse(result)

@app.get("/")
def home():
    return {"message": "Document Tamper Screening Service is running"}
