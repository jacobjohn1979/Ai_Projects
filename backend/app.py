"""
app.py — Banking Document Fraud Detection API v3.1

Response strategy:
  POST /screen-pdf      → sync  (~2-5s)  → full result immediately
  POST /screen-image    → sync  (~3-8s)  → full result immediately
  POST /screen-id-card  → async (~15-30s)→ task_id + optional webhook callback
  GET  /result/{id}     → poll for id card result
  GET  /history/{id}    → submission history
"""

import os
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from celery_app import celery
from database   import init_db, get_submission_history, save_screening_log, check_velocity
from screening  import (
    # PDF
    inspect_pdf_forensics, extract_text_and_layout,
    run_ocr_if_needed, field_checks_pdf, score_pdf,
    # Image
    extract_image_metadata, perform_ela, noise_analysis,
    copy_move_detection, basic_image_forensics, score_image,
    # Shared
    DATE_REGEX, ID_REGEX,
)

import re
import pytesseract
from PIL import Image as PILImage

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("fraud_detect")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_upload(file_bytes: bytes, filename: str) -> str:
    file_id   = uuid.uuid4().hex
    save_path = str(UPLOAD_DIR / f"{file_id}_{filename}")
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    return save_path


def _remove(p):
    try: Path(p).unlink(missing_ok=True)
    except Exception: pass


def _risk_summary(risk_score: int, risk_level: str, flags: list) -> dict:
    """Human-readable risk summary for API consumers."""
    descriptions = {
        "HIGH":   "Document shows strong indicators of tampering or fraud. Manual review required before proceeding.",
        "MEDIUM": "Document has some anomalies. Recommend additional verification steps.",
        "LOW":    "No significant tampering indicators detected. Document appears authentic.",
    }
    return {
        "level":       risk_level,
        "score":       risk_score,
        "description": descriptions.get(risk_level, ""),
        "flag_count":  len(flags),
        "action":      "REJECT" if risk_level == "HIGH" else (
                       "REVIEW" if risk_level == "MEDIUM" else "PASS"),
    }


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Banking Document Fraud Detection API",
    description="""
## Banking KYC / Onboarding / Loan Document Fraud Screening

### Response Strategy
| Endpoint | Mode | Typical time |
|---|---|---|
| `/screen-pdf` | **Synchronous** — full result returned immediately | 2–5 sec |
| `/screen-image` | **Synchronous** — full result returned immediately | 3–8 sec |
| `/screen-id-card` | **Async** — returns `task_id`, fires webhook when done | 15–30 sec |

### Risk Levels
- **LOW** → `action: PASS` — proceed with onboarding
- **MEDIUM** → `action: REVIEW` — request additional verification  
- **HIGH** → `action: REJECT` — flag for manual review
    """,
    version="3.1.0",
    on_startup=[init_db],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF — SYNCHRONOUS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/screen-pdf",
    summary="Screen a PDF document for tampering",
    response_description="Full fraud screening result with risk rating",
)
async def screen_pdf(
    file:         UploadFile = File(..., description="PDF file to screen"),
    applicant_id: str | None = Form(None, description="Your internal applicant/case reference"),
):
    """
    Synchronous PDF screening. Returns the full result immediately.

    Checks performed:
    - Metadata forensics (creator, producer, timestamps)
    - Font diversity analysis (pasted content detection)
    - JavaScript / redaction detection
    - OCR for scanned PDFs
    - Amount and date field validation
    - Velocity checks (duplicate/rapid resubmission)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    save_path = _save_upload(await file.read(), file.filename)

    try:
        sha256   = _sha256(save_path)
        velocity = check_velocity(None, sha256)

        forensic_info, forensic_flags = inspect_pdf_forensics(save_path)
        text_info,     text_flags     = extract_text_and_layout(save_path)

        scanned_like = ("image_based_pdf" in text_flags or
                        "mixed_digital_and_scanned_pages" in text_flags)

        ocr_info, ocr_flags = run_ocr_if_needed(save_path, scanned_like)

        final_text = text_info["text"] or ocr_info.get("ocr_text", "")
        field_info, field_flags = field_checks_pdf(final_text)

        all_flags  = forensic_flags + text_flags + ocr_flags + field_flags + velocity["flags"]
        risk_score, risk_level = score_pdf(all_flags)

        result = {
            # ── Identity ──────────────────────────────────────────────────────
            "file_name":    file.filename,
            "sha256":       sha256,
            "screened_at":  datetime.utcnow().isoformat() + "Z",
            "applicant_id": applicant_id,
            "document_type": "pdf",

            # ── Risk decision ─────────────────────────────────────────────────
            "risk": _risk_summary(risk_score, risk_level, all_flags),

            # ── Flags ─────────────────────────────────────────────────────────
            "flags": all_flags,

            # ── Detail sections ───────────────────────────────────────────────
            "forensics": forensic_info,
            "text_summary": {
                "total_pages":        text_info["total_pages"],
                "scanned_like_pages": text_info["scanned_like_pages"],
                "text_length":        len(final_text),
                "fonts":              text_info.get("fonts", {}),
            },
            "field_checks": field_info,
            "velocity":     velocity["counts"],
            "ocr_used":     ocr_info.get("ocr_used", False),
        }

        save_screening_log(result, file.filename, "pdf", "pdf", applicant_id)
        return JSONResponse(result)

    finally:
        _remove(save_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE — SYNCHRONOUS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/screen-image",
    summary="Screen a document image for forgery",
    response_description="Full fraud screening result with risk rating",
)
async def screen_image(
    file:         UploadFile = File(..., description="Image file to screen (JPG, PNG, BMP, TIFF)"),
    applicant_id: str | None = Form(None, description="Your internal applicant/case reference"),
):
    """
    Synchronous image screening. Returns the full result immediately.

    Checks performed:
    - EXIF metadata analysis (editing software detection)
    - Error Level Analysis (ELA) — detects composited/edited regions
    - Noise consistency analysis — detects spliced images
    - Copy-move detection (SIFT) — detects cloned regions
    - Double JPEG compression detection
    - OCR field validation
    - Velocity checks
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Accepted: {', '.join(IMAGE_EXTS)}")

    save_path = _save_upload(await file.read(), file.filename)

    try:
        sha256   = _sha256(save_path)
        velocity = check_velocity(None, sha256)

        metadata_info, metadata_flags = extract_image_metadata(save_path)
        ela_info,      ela_flags      = perform_ela(save_path)
        noise_info,    noise_flags    = noise_analysis(save_path)
        cm_info,       cm_flags       = copy_move_detection(save_path)
        forensic_info, forensic_flags = basic_image_forensics(save_path)

        ocr_text  = ""
        ocr_flags = []
        try:
            ocr_text = pytesseract.image_to_string(PILImage.open(save_path))
            if len(ocr_text.strip()) < 10:
                ocr_flags.append("very_low_ocr_text")
            if not re.findall(DATE_REGEX, ocr_text):
                ocr_flags.append("no_date_pattern_found")
            if not re.findall(ID_REGEX, ocr_text):
                ocr_flags.append("no_id_like_pattern_found")
        except Exception:
            ocr_flags.append("ocr_failed")

        all_flags  = (metadata_flags + ela_flags + noise_flags + cm_flags +
                      forensic_flags + ocr_flags + velocity["flags"])
        risk_score, risk_level = score_image(all_flags)

        result = {
            "file_name":    file.filename,
            "sha256":       sha256,
            "screened_at":  datetime.utcnow().isoformat() + "Z",
            "applicant_id": applicant_id,
            "document_type": "image",

            "risk": _risk_summary(risk_score, risk_level, all_flags),

            "flags": all_flags,

            "metadata":        metadata_info,
            "ela":             ela_info,
            "noise_analysis":  noise_info,
            "copy_move":       cm_info,
            "image_forensics": forensic_info,
            "velocity":        velocity["counts"],
            "ocr_summary": {
                "text_length":  len(ocr_text),
                "text_preview": ocr_text[:400],
            },
        }

        save_screening_log(result, file.filename, "image", "image", applicant_id)
        return JSONResponse(result)

    finally:
        _remove(save_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  ID CARD — ASYNC + WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/screen-id-card",
    summary="Full forensic ID card / passport screening (async)",
    status_code=202,
    response_description="Task ID for polling, webhook fired on completion",
)
async def screen_id_card(
    file:         UploadFile = File(...,  description="ID card or passport image"),
    selfie:       UploadFile | None = File(None, description="Applicant selfie for face match (optional)"),
    applicant_id: str | None = Form(None, description="Your internal applicant/case reference"),
    callback_url: str | None = Form(None, description="Webhook URL to receive result when screening completes"),
):
    """
    Async ID card screening — returns immediately with a `task_id`.
    Full result delivered via webhook to `callback_url` when complete (15–30 sec).
    Also available via `GET /result/{task_id}` for polling.

    Checks performed:
    - EXIF metadata + editing software detection
    - Error Level Analysis (ELA)
    - Noise consistency + copy-move detection
    - MRZ extraction + ICAO 9303 checksum validation
    - Expiry date validation
    - Hologram / security feature detection
    - Country template matching (aspect ratio, zones, keywords)
    - Face liveness analysis (if selfie provided)
    - Face match selfie vs ID photo (if selfie provided)
    - Velocity checks (duplicate ID, prior HIGH-risk submissions)
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Accepted: {', '.join(IMAGE_EXTS)}")

    save_path   = _save_upload(await file.read(), file.filename)
    selfie_path = _save_upload(await selfie.read(), selfie.filename) if selfie else None

    from tasks import screen_id_card_task
    task = screen_id_card_task.apply_async(
        args=[save_path, file.filename, selfie_path, applicant_id, callback_url]
    )

    return JSONResponse({
        "task_id":      task.id,
        "status":       "queued",
        "applicant_id": applicant_id,
        "file_name":    file.filename,
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "webhook":      callback_url or None,
        "poll_url":     f"/result/{task.id}",
        "message":      "Screening in progress. Poll poll_url or await webhook callback.",
    }, status_code=202)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT POLLING + HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/result/{task_id}",
    summary="Poll for ID card screening result",
)
def get_result(task_id: str):
    """Poll for the result of an async ID card screening task."""
    result = celery.AsyncResult(task_id)

    if result.state == "PENDING":
        return JSONResponse({"task_id": task_id, "status": "pending",
                             "message": "Task is queued, not yet started."})
    if result.state == "STARTED":
        return JSONResponse({"task_id": task_id, "status": "processing",
                             "message": "Screening is in progress."})
    if result.state == "FAILURE":
        return JSONResponse({"task_id": task_id, "status": "failed",
                             "error": str(result.result),
                             "message": "Screening failed. Please resubmit."
                             }, status_code=500)
    if result.state == "SUCCESS":
        data = result.result
        return JSONResponse({
            "task_id": task_id,
            "status":  "complete",
            "result":  data,
        })

    return JSONResponse({"task_id": task_id, "status": result.state})


@app.get(
    "/history/{id_number}",
    summary="Submission history for a given ID number",
)
def submission_history(id_number: str, limit: int = 20):
    """Returns past screening results for an ID number — useful for velocity/repeat checks."""
    return JSONResponse(get_submission_history(id_number, limit))


@app.get("/", summary="Health check", include_in_schema=False)
def home():
    return {
        "service":    "Banking Document Fraud Detection API",
        "version":    "3.1.0",
        "status":     "running",
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }
