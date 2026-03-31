"""
tasks.py — Celery async tasks
Only ID card screening runs as a Celery task (PDF and image are now sync in app.py).
Fires a webhook callback on completion if callback_url is provided.
"""
import os
import re
import json
import uuid
import hashlib
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

from celery_app import celery
from database   import save_screening_log, check_velocity
from screening  import (
    extract_image_metadata, perform_ela, noise_analysis,
    copy_move_detection, extract_id_card_fields,
    analyze_id_card_regions, score_id_card,
)

log = logging.getLogger("fraud_detect.tasks")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "10"))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _remove(path):
    try: Path(path).unlink(missing_ok=True)
    except Exception: pass


def _fire_webhook(callback_url: str, payload: dict):
    """
    POST the screening result to the callback URL.
    Uses stdlib only — no extra dependencies.
    Fails silently so a bad webhook never breaks the task result.
    """
    if not callback_url:
        return

    try:
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url     = callback_url,
            data    = body,
            method  = "POST",
            headers = {
                "Content-Type":  "application/json",
                "User-Agent":    "FraudDetect-Webhook/3.1",
                "X-Event-Type":  "screening.complete",
                "X-Risk-Level":  payload.get("risk", {}).get("level", ""),
                "X-Action":      payload.get("risk", {}).get("action", ""),
            },
        )
        with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT) as resp:
            log.info(f"Webhook delivered to {callback_url} — HTTP {resp.status}")

    except urllib.error.HTTPError as e:
        log.warning(f"Webhook HTTP error {e.code} for {callback_url}")
    except urllib.error.URLError as e:
        log.warning(f"Webhook delivery failed for {callback_url}: {e.reason}")
    except Exception as e:
        log.warning(f"Webhook unexpected error for {callback_url}: {e}")


def _risk_summary(risk_score: int, risk_level: str, flags: list) -> dict:
    descriptions = {
        "HIGH":   "Document shows strong indicators of tampering or fraud. Manual review required.",
        "MEDIUM": "Document has some anomalies. Recommend additional verification.",
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ID CARD TASK — async, fires webhook on completion
# ═══════════════════════════════════════════════════════════════════════════════

@celery.task(bind=True, name="tasks.screen_id_card_task", max_retries=2, default_retry_delay=10)
def screen_id_card_task(
    self,
    file_path:    str,
    filename:     str,
    selfie_path:  str | None = None,
    applicant_id: str | None = None,
    callback_url: str | None = None,
):
    try:
        from face_match     import analyze_liveness, match_faces
        from hologram       import analyze_hologram
        from template_match import match_template

        sha256   = _sha256(file_path)
        velocity = check_velocity(None, sha256)

        # ── Core forensics ────────────────────────────────────────────────────
        metadata_info, metadata_flags = extract_image_metadata(file_path)
        ela_info,      ela_flags      = perform_ela(file_path)
        noise_info,    noise_flags    = noise_analysis(file_path)
        cm_info,       cm_flags       = copy_move_detection(file_path)
        field_info,    field_flags    = extract_id_card_fields(file_path)
        region_info,   region_flags   = analyze_id_card_regions(file_path)

        id_number = field_info.get("id_number")
        mrz_lines = field_info.get("mrz_lines", [])
        ocr_text  = field_info.get("ocr_text", "")

        # ── Velocity check with ID number ─────────────────────────────────────
        if id_number:
            id_velocity = check_velocity(id_number, sha256)
            velocity["flags"]  += id_velocity["flags"]
            velocity["counts"].update(id_velocity["counts"])

        # ── Hologram + template ───────────────────────────────────────────────
        holo_info, holo_flags = analyze_hologram(file_path)
        tmpl_info, tmpl_flags = match_template(file_path, ocr_text, mrz_lines)

        # ── Face liveness + match ─────────────────────────────────────────────
        liveness_info,   liveness_flags   = {}, []
        face_match_info, face_match_flags = {}, []

        if selfie_path and Path(selfie_path).exists():
            liveness_info,   liveness_flags   = analyze_liveness(selfie_path)
            face_match_info, face_match_flags = match_faces(selfie_path, file_path)
        else:
            liveness_flags   = ["selfie_not_provided"]
            face_match_flags = ["selfie_not_provided"]

        # ── Score ─────────────────────────────────────────────────────────────
        all_flags = (
            metadata_flags + ela_flags + noise_flags + cm_flags +
            field_flags + region_flags + holo_flags + tmpl_flags +
            liveness_flags + face_match_flags + velocity["flags"]
        )
        risk_score, risk_level = score_id_card(all_flags)

        # ── Build result ──────────────────────────────────────────────────────
        result = {
            "task_id":       self.request.id,
            "file_name":     filename,
            "sha256":        sha256,
            "screened_at":   datetime.utcnow().isoformat() + "Z",
            "applicant_id":  applicant_id,
            "document_type": "id_card",

            # ── Risk decision (top-level for easy parsing) ────────────────────
            "risk": _risk_summary(risk_score, risk_level, all_flags),

            "flags": all_flags,

            # ── Detail sections ───────────────────────────────────────────────
            "field_info": {
                "id_number":       id_number,
                "dob":             field_info.get("dob"),
                "expiry_date":     field_info.get("expiry_date"),
                "mrz_lines":       mrz_lines,
                "mrz_checksum_ok": field_info.get("mrz_dob_checksum_ok"),
                "text_length":     field_info.get("text_length", 0),
            },
            "metadata":       metadata_info,
            "ela":            ela_info,
            "noise_analysis": noise_info,
            "copy_move":      cm_info,
            "region_info":    region_info,
            "hologram":       holo_info,
            "template_match": tmpl_info,
            "liveness":       liveness_info,
            "face_match":     face_match_info,
            "velocity":       velocity["counts"],
        }

        # ── Save to DB ────────────────────────────────────────────────────────
        save_screening_log(result, filename, "image", "id_card", applicant_id)

        # ── Fire webhook ──────────────────────────────────────────────────────
        if callback_url:
            _fire_webhook(callback_url, {
                "event":        "screening.complete",
                "task_id":      self.request.id,
                "applicant_id": applicant_id,
                "file_name":    filename,
                "screened_at":  result["screened_at"],
                "risk":         result["risk"],       # top-level risk for easy parsing
                "flags":        all_flags,
                "detail":       result,               # full detail for systems that need it
            })

        return result

    except Exception as exc:
        log.error(f"screen_id_card_task failed: {exc}")

        # ── Notify webhook of failure too ─────────────────────────────────────
        if callback_url:
            _fire_webhook(callback_url, {
                "event":        "screening.failed",
                "task_id":      self.request.id,
                "applicant_id": applicant_id,
                "file_name":    filename,
                "error":        str(exc),
            })

        raise self.retry(exc=exc)

    finally:
        _remove(file_path)
        if selfie_path:
            _remove(selfie_path)
