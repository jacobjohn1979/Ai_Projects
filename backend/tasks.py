"""
tasks.py — Celery async tasks for document screening
Each task mirrors an endpoint but runs in a background worker.
Results are stored in Redis and retrievable via task ID.
"""
import os
import re
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime

from celery_app import celery
from database import save_screening_log, check_velocity

log = logging.getLogger("fraud_detect.tasks")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATE_REGEX = r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"
ID_REGEX   = r"\b[A-Z0-9]{6,20}\b"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _remove(path):
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF TASK
# ═══════════════════════════════════════════════════════════════════════════════

@celery.task(bind=True, name="tasks.screen_pdf_task", max_retries=2, default_retry_delay=5)
def screen_pdf_task(self, file_path: str, filename: str, applicant_id: str | None = None):
    """
    Async PDF screening task.
    file_path: absolute path to the already-saved upload.
    """
    try:
        # Import here to avoid circular imports at module load
        from app import (
            inspect_pdf_forensics, extract_text_and_layout,
            run_ocr_if_needed, field_checks_pdf, score_pdf,
        )

        sha256 = _sha256(file_path)

        # velocity check on file hash
        velocity = check_velocity(None, sha256)
        velocity_flags = velocity["flags"]

        forensic_info, forensic_flags = inspect_pdf_forensics(file_path)
        text_info,     text_flags     = extract_text_and_layout(file_path)

        scanned_like = ("image_based_pdf" in text_flags or
                        "mixed_digital_and_scanned_pages" in text_flags)

        ocr_info, ocr_flags = run_ocr_if_needed(file_path, scanned_like)

        final_text = text_info["text"] or ocr_info.get("ocr_text", "")
        field_info, field_flags = field_checks_pdf(final_text)

        all_flags  = forensic_flags + text_flags + ocr_flags + field_flags + velocity_flags
        risk_score, risk_level = score_pdf(all_flags)

        result = {
            "task_id":      self.request.id,
            "file_name":    filename,
            "sha256":       sha256,
            "screened_at":  datetime.utcnow().isoformat(),
            "applicant_id": applicant_id,
            "risk_score":   risk_score,
            "risk_level":   risk_level,
            "flags":        all_flags,
            "velocity":     velocity["counts"],
            "forensics":    forensic_info,
            "text_summary": {
                "total_pages":        text_info["total_pages"],
                "scanned_like_pages": text_info["scanned_like_pages"],
                "text_length":        len(final_text),
                "fonts":              text_info.get("fonts", {}),
            },
            "field_checks": field_info,
            "ocr":          {"used": ocr_info.get("ocr_used", False)},
        }

        save_screening_log(result, filename, "pdf", "pdf", applicant_id)
        return result

    except Exception as exc:
        log.error(f"screen_pdf_task failed: {exc}")
        raise self.retry(exc=exc)
    finally:
        _remove(file_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE TASK
# ═══════════════════════════════════════════════════════════════════════════════

@celery.task(bind=True, name="tasks.screen_image_task", max_retries=2, default_retry_delay=5)
def screen_image_task(self, file_path: str, filename: str, applicant_id: str | None = None):
    try:
        import pytesseract
        from PIL import Image
        from app import (
            extract_image_metadata, perform_ela,
            noise_analysis, copy_move_detection, basic_image_forensics,
        )

        sha256 = _sha256(file_path)
        velocity = check_velocity(None, sha256)

        metadata_info, metadata_flags = extract_image_metadata(file_path)
        ela_info,      ela_flags      = perform_ela(file_path)
        noise_info,    noise_flags    = noise_analysis(file_path)
        cm_info,       cm_flags       = copy_move_detection(file_path)
        forensic_info, forensic_flags = basic_image_forensics(file_path)

        ocr_text = ""
        ocr_flags = []
        try:
            ocr_text = pytesseract.image_to_string(Image.open(file_path))
            if len(ocr_text.strip()) < 10:
                ocr_flags.append("very_low_ocr_text")
            if not re.findall(DATE_REGEX, ocr_text):
                ocr_flags.append("no_date_pattern_found")
            if not re.findall(ID_REGEX, ocr_text):
                ocr_flags.append("no_id_like_pattern_found")
        except Exception:
            ocr_flags.append("ocr_failed")

        from app import score_image
        all_flags  = (metadata_flags + ela_flags + noise_flags + cm_flags +
                      forensic_flags + ocr_flags + velocity["flags"])
        risk_score, risk_level = score_image(all_flags)

        result = {
            "task_id":        self.request.id,
            "file_name":      filename,
            "sha256":         sha256,
            "screened_at":    datetime.utcnow().isoformat(),
            "applicant_id":   applicant_id,
            "risk_score":     risk_score,
            "risk_level":     risk_level,
            "flags":          all_flags,
            "velocity":       velocity["counts"],
            "metadata":       metadata_info,
            "ela":            ela_info,
            "noise_analysis": noise_info,
            "copy_move":      cm_info,
            "image_forensics": forensic_info,
            "ocr_summary":    {"text_length": len(ocr_text), "text_preview": ocr_text[:400]},
        }

        save_screening_log(result, filename, "image", "image", applicant_id)
        return result

    except Exception as exc:
        log.error(f"screen_image_task failed: {exc}")
        raise self.retry(exc=exc)
    finally:
        _remove(file_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  ID CARD TASK  (heaviest — face + hologram + template + forensics)
# ═══════════════════════════════════════════════════════════════════════════════

@celery.task(bind=True, name="tasks.screen_id_card_task", max_retries=2, default_retry_delay=10)
def screen_id_card_task(
    self,
    file_path:   str,
    filename:    str,
    selfie_path: str | None = None,
    applicant_id: str | None = None,
):
    try:
        from app import (
            extract_image_metadata, perform_ela,
            noise_analysis, copy_move_detection,
            extract_id_card_fields, analyze_id_card_regions,
            score_id_card,
        )
        from face_match   import analyze_liveness, match_faces
        from hologram     import analyze_hologram
        from template_match import match_template

        sha256   = _sha256(file_path)
        id_number = None

        # pre-check velocity on file hash
        velocity = check_velocity(None, sha256)

        # ── core forensics ────────────────────────────────────────────────────
        metadata_info, metadata_flags = extract_image_metadata(file_path)
        ela_info,      ela_flags      = perform_ela(file_path)
        noise_info,    noise_flags    = noise_analysis(file_path)
        cm_info,       cm_flags       = copy_move_detection(file_path)
        field_info,    field_flags    = extract_id_card_fields(file_path)
        region_info,   region_flags   = analyze_id_card_regions(file_path)

        id_number  = field_info.get("id_number")
        mrz_lines  = field_info.get("mrz_lines", [])
        ocr_text   = field_info.get("ocr_text", "")

        # ── velocity check with ID number ─────────────────────────────────────
        if id_number:
            id_velocity = check_velocity(id_number, sha256)
            velocity["flags"]  += id_velocity["flags"]
            velocity["counts"].update(id_velocity["counts"])

        # ── hologram analysis ─────────────────────────────────────────────────
        holo_info, holo_flags = analyze_hologram(file_path)

        # ── template matching ─────────────────────────────────────────────────
        tmpl_info, tmpl_flags = match_template(file_path, ocr_text, mrz_lines)

        # ── liveness on selfie (if provided) ──────────────────────────────────
        liveness_info, liveness_flags = {}, []
        face_match_info, face_match_flags = {}, []

        if selfie_path and Path(selfie_path).exists():
            liveness_info,   liveness_flags   = analyze_liveness(selfie_path)
            face_match_info, face_match_flags = match_faces(selfie_path, file_path)
        else:
            liveness_flags   = ["selfie_not_provided"]
            face_match_flags = ["selfie_not_provided"]

        all_flags = (
            metadata_flags + ela_flags + noise_flags + cm_flags +
            field_flags + region_flags + holo_flags + tmpl_flags +
            liveness_flags + face_match_flags + velocity["flags"]
        )

        risk_score, risk_level = score_id_card(all_flags)

        result = {
            "task_id":      self.request.id,
            "file_name":    filename,
            "sha256":       sha256,
            "screened_at":  datetime.utcnow().isoformat(),
            "applicant_id": applicant_id,
            "risk_score":   risk_score,
            "risk_level":   risk_level,
            "flags":        all_flags,
            "velocity":     velocity["counts"],
            "metadata":     metadata_info,
            "ela":          ela_info,
            "noise_analysis": noise_info,
            "copy_move":    cm_info,
            "field_info": {
                "id_number":       id_number,
                "dob":             field_info.get("dob"),
                "expiry_date":     field_info.get("expiry_date"),
                "mrz_lines":       mrz_lines,
                "mrz_checksum_ok": field_info.get("mrz_dob_checksum_ok"),
                "text_length":     field_info.get("text_length", 0),
            },
            "region_info":    region_info,
            "hologram":       holo_info,
            "template_match": tmpl_info,
            "liveness":       liveness_info,
            "face_match":     face_match_info,
        }

        save_screening_log(result, filename, "image", "id_card", applicant_id)
        return result

    except Exception as exc:
        log.error(f"screen_id_card_task failed: {exc}")
        raise self.retry(exc=exc)
    finally:
        _remove(file_path)
        if selfie_path:
            _remove(selfie_path)
