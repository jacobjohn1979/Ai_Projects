"""
duplicate_detect.py — Duplicate Identity Detection
Flags when the same person submits documents under different applicant names.
Detection methods:
  1. Same ID number, different applicant_id
  2. Same file hash (exact duplicate)
  3. Similar face embedding (if DeepFace available)
"""
import os
import json
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("fraud_detect.duplicate")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def check_duplicate_id(id_number: str, applicant_id: str) -> dict:
    """
    Check if same ID number was submitted under a different applicant name.
    Returns dict with findings.
    """
    if not id_number:
        return {"checked": False, "reason": "no_id_number"}

    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT applicant_id, file_name, risk_level, screened_at, id
            FROM screening_logs
            WHERE id_number = :id
            AND applicant_id IS NOT NULL
            AND applicant_id != :app
            ORDER BY screened_at DESC
            LIMIT 5
        """), {"id": id_number, "app": applicant_id or ""}).fetchall()

        if not rows:
            return {"checked": True, "duplicate_found": False}

        matches = [dict(r._mapping) for r in rows]
        for m in matches:
            if m.get("screened_at"):
                m["screened_at"] = m["screened_at"].isoformat()

        return {
            "checked":          True,
            "duplicate_found":  True,
            "same_id_different_applicant": True,
            "matches":          matches,
            "match_count":      len(matches),
            "flag":             "duplicate_id_different_applicant",
        }
    finally:
        db.close()


def check_duplicate_file(sha256: str, applicant_id: str) -> dict:
    """Check if exact same file was submitted before under different applicant."""
    if not sha256:
        return {"checked": False}

    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT applicant_id, file_name, risk_level, screened_at, id
            FROM screening_logs
            WHERE file_sha256 = :h
            AND (applicant_id IS NULL OR applicant_id != :app)
            ORDER BY screened_at DESC
            LIMIT 5
        """), {"h": sha256, "app": applicant_id or ""}).fetchall()

        if not rows:
            return {"checked": True, "duplicate_found": False}

        matches = [dict(r._mapping) for r in rows]
        for m in matches:
            if m.get("screened_at"):
                m["screened_at"] = m["screened_at"].isoformat()

        return {
            "checked":         True,
            "duplicate_found": True,
            "same_file":       True,
            "matches":         matches,
            "flag":            "duplicate_file_different_applicant",
        }
    finally:
        db.close()


def get_all_duplicates(days: int = 90) -> list:
    """
    Find all duplicate identity cases in the database.
    Returns list of groups where same ID number appears under different applicants.
    """
    since = datetime.utcnow() - timedelta(days=days)
    db    = SessionLocal()
    try:
        # Find ID numbers that appear under multiple applicant IDs
        rows = db.execute(text("""
            SELECT id_number,
                   COUNT(DISTINCT applicant_id) AS applicant_count,
                   ARRAY_AGG(DISTINCT applicant_id) AS applicants,
                   COUNT(*) AS submission_count,
                   MAX(risk_level) AS max_risk,
                   MAX(screened_at) AS last_seen
            FROM screening_logs
            WHERE id_number IS NOT NULL
            AND applicant_id IS NOT NULL
            AND screened_at >= :since
            GROUP BY id_number
            HAVING COUNT(DISTINCT applicant_id) > 1
            ORDER BY applicant_count DESC, last_seen DESC
            LIMIT 50
        """), {"since": since}).fetchall()

        results = []
        for r in rows:
            row = dict(r._mapping)
            if row.get("last_seen"):
                row["last_seen"] = row["last_seen"].isoformat()
            # get full records for this ID number
            records = db.execute(text("""
                SELECT id, file_name, applicant_id, risk_level, risk_score,
                       screened_at, doc_type
                FROM screening_logs
                WHERE id_number = :id AND screened_at >= :since
                ORDER BY screened_at DESC
            """), {"id": row["id_number"], "since": since}).fetchall()

            row["records"] = []
            for rec in records:
                rd = dict(rec._mapping)
                if rd.get("screened_at"):
                    rd["screened_at"] = rd["screened_at"].isoformat()
                row["records"].append(rd)

            results.append(row)

        return results
    finally:
        db.close()


def get_duplicate_stats(days: int = 30) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    db    = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT COUNT(*) AS duplicate_groups
            FROM (
                SELECT id_number
                FROM screening_logs
                WHERE id_number IS NOT NULL
                AND applicant_id IS NOT NULL
                AND screened_at >= :since
                GROUP BY id_number
                HAVING COUNT(DISTINCT applicant_id) > 1
            ) sub
        """), {"since": since}).fetchone()
        return {"duplicate_groups": result.duplicate_groups if result else 0}
    finally:
        db.close()
