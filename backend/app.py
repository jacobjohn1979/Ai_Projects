"""
app.py — Banking Document Fraud Detection API v3.0
FastAPI entry point. All screening logic lives in screening.py.
Endpoints submit jobs to Celery workers and return a task_id immediately.
Poll /result/{task_id} to retrieve the completed screening report.
"""

import os
import uuid
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from celery_app import celery
from database   import init_db, get_submission_history

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("fraud_detect")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_upload(file_bytes: bytes, filename: str) -> str:
    file_id   = uuid.uuid4().hex
    save_path = str(UPLOAD_DIR / f"{file_id}_{filename}")
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    return save_path


def _submit(task_fn, *args, **kwargs) -> JSONResponse:
    task = task_fn.apply_async(args=args, kwargs=kwargs)
    return JSONResponse({"task_id": task.id, "status": "queued"}, status_code=202)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Banking Document Fraud Detection API",
    description="KYC / Onboarding / Loan — full async fraud screening",
    version="3.0.0",
    on_startup=[init_db],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/screen-pdf", summary="Submit a PDF for async fraud screening", status_code=202)
async def screen_pdf(
    file:         UploadFile = File(...),
    applicant_id: str | None = Form(None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    save_path = _save_upload(await file.read(), file.filename)
    from tasks import screen_pdf_task
    return _submit(screen_pdf_task, save_path, file.filename, applicant_id)


@app.post("/screen-image", summary="Submit a document image for async fraud screening", status_code=202)
async def screen_image(
    file:         UploadFile = File(...),
    applicant_id: str | None = Form(None),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported image type: {ext}")
    save_path = _save_upload(await file.read(), file.filename)
    from tasks import screen_image_task
    return _submit(screen_image_task, save_path, file.filename, applicant_id)


@app.post("/screen-id-card", summary="Full forensic ID card screening", status_code=202)
async def screen_id_card(
    file:         UploadFile = File(...),
    selfie:       UploadFile | None = File(None),
    applicant_id: str | None = Form(None),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported image type: {ext}")
    save_path   = _save_upload(await file.read(), file.filename)
    selfie_path = _save_upload(await selfie.read(), selfie.filename) if selfie else None
    from tasks import screen_id_card_task
    return _submit(screen_id_card_task, save_path, file.filename, selfie_path, applicant_id)


@app.get("/result/{task_id}", summary="Poll for async screening result")
def get_result(task_id: str):
    result = celery.AsyncResult(task_id)
    if result.state == "PENDING":
        return JSONResponse({"task_id": task_id, "status": "pending"})
    if result.state == "STARTED":
        return JSONResponse({"task_id": task_id, "status": "processing"})
    if result.state == "FAILURE":
        return JSONResponse({"task_id": task_id, "status": "failed",
                             "error": str(result.result)}, status_code=500)
    if result.state == "SUCCESS":
        return JSONResponse({"task_id": task_id, "status": "complete",
                             "result": result.result})
    return JSONResponse({"task_id": task_id, "status": result.state})


@app.get("/history/{id_number}", summary="Submission history for an ID number")
def submission_history(id_number: str, limit: int = 20):
    return JSONResponse(get_submission_history(id_number, limit))


@app.get("/", summary="Health check")
def home():
    return {
        "service":   "Banking Document Fraud Detection API v3.0",
        "status":    "running",
        "screened_at": datetime.utcnow().isoformat(),
        "endpoints": [
            "POST /screen-pdf",
            "POST /screen-image",
            "POST /screen-id-card",
            "GET  /result/{task_id}",
            "GET  /history/{id_number}",
        ],
    }
