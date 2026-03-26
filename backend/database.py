"""
database.py — PostgreSQL via SQLAlchemy
  - screening_logs  : audit trail for every screened document
  - velocity_checks : detect duplicate / rapid re-submission of same ID
"""
import os
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    DateTime, JSON, Text, Index, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

load_dotenv()
log = logging.getLogger("fraud_detect.db")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/fraud_detect"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── Models ─────────────────────────────────────────────────────────────────────

class ScreeningLog(Base):
    __tablename__ = "screening_logs"

    id           = Column(Integer, primary_key=True, index=True)
    file_name    = Column(String(255))
    file_sha256  = Column(String(64), index=True)
    category     = Column(String(50))   # pdf | image | id_card
    doc_type     = Column(String(50))
    risk_score   = Column(Integer)
    risk_level   = Column(String(10))   # LOW | MEDIUM | HIGH
    flags        = Column(JSON)
    full_result  = Column(JSON)
    screened_at  = Column(DateTime, default=datetime.utcnow, index=True)
    id_number    = Column(String(50), nullable=True, index=True)
    applicant_id = Column(String(100), nullable=True, index=True)


class VelocityEvent(Base):
    __tablename__ = "velocity_events"

    id           = Column(Integer, primary_key=True, index=True)
    id_number    = Column(String(50), index=True)
    file_sha256  = Column(String(64), index=True)
    applicant_id = Column(String(100), nullable=True)
    risk_level   = Column(String(10))
    submitted_at = Column(DateTime, default=datetime.utcnow, index=True)


# Composite indexes for velocity queries
Index("ix_velocity_id_time",   VelocityEvent.id_number,   VelocityEvent.submitted_at)
Index("ix_velocity_hash_time", VelocityEvent.file_sha256, VelocityEvent.submitted_at)


def init_db():
    """Create tables if they don't exist — safe to call multiple times."""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        log.info("Database tables initialised")
    except Exception as e:
        log.error(f"DB init failed (non-fatal): {e}")


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Screening log ──────────────────────────────────────────────────────────────

def save_screening_log(
    result: dict,
    filename: str,
    category: str,
    doc_type: str,
    applicant_id: str | None = None,
):
    db = SessionLocal()
    try:
        id_number = None
        if doc_type == "id_card":
            id_number = result.get("field_info", {}).get("id_number")

        log_entry = ScreeningLog(
            file_name    = filename,
            file_sha256  = result.get("sha256"),
            category     = category,
            doc_type     = doc_type,
            risk_score   = result.get("risk_score", 0),
            risk_level   = result.get("risk_level", "UNKNOWN"),
            flags        = result.get("flags", []),
            full_result  = result,
            id_number    = id_number,
            applicant_id = applicant_id,
        )
        db.add(log_entry)

        # also record velocity event for ID cards
        if id_number:
            ev = VelocityEvent(
                id_number    = id_number,
                file_sha256  = result.get("sha256"),
                applicant_id = applicant_id,
                risk_level   = result.get("risk_level", "UNKNOWN"),
            )
            db.add(ev)

        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"save_screening_log failed: {e}")
    finally:
        db.close()


# ── Velocity checks ────────────────────────────────────────────────────────────

VELOCITY_WINDOW_HOURS = int(os.getenv("VELOCITY_WINDOW_HOURS", "24"))
MAX_SUBMISSIONS_PER_ID = int(os.getenv("MAX_SUBMISSIONS_PER_ID", "3"))
MAX_SAME_FILE_SUBMISSIONS = int(os.getenv("MAX_SAME_FILE_SUBMISSIONS", "2"))


def check_velocity(id_number: str | None, file_sha256: str) -> dict:
    """
    Returns velocity flags and counts for a given ID number / file hash.
    Call this BEFORE saving the new screening log.
    """
    flags  = []
    counts = {}
    db     = SessionLocal()
    window_start = datetime.utcnow() - timedelta(hours=VELOCITY_WINDOW_HOURS)

    try:
        # ── 1. Same file hash submitted before (exact duplicate) ────────────
        hash_count = (
            db.query(func.count(VelocityEvent.id))
            .filter(
                VelocityEvent.file_sha256 == file_sha256,
                VelocityEvent.submitted_at >= window_start,
            )
            .scalar()
        )
        counts["same_file_submissions_24h"] = hash_count
        if hash_count >= MAX_SAME_FILE_SUBMISSIONS:
            flags.append("duplicate_file_resubmission")

        # ── 2. Same ID number submitted too many times ───────────────────────
        if id_number:
            id_count = (
                db.query(func.count(VelocityEvent.id))
                .filter(
                    VelocityEvent.id_number == id_number,
                    VelocityEvent.submitted_at >= window_start,
                )
                .scalar()
            )
            counts["same_id_submissions_24h"] = id_count
            if id_count >= MAX_SUBMISSIONS_PER_ID:
                flags.append("id_number_velocity_breach")

            # ── 3. Same ID previously flagged HIGH risk ──────────────────────
            prior_high = (
                db.query(func.count(VelocityEvent.id))
                .filter(
                    VelocityEvent.id_number == id_number,
                    VelocityEvent.risk_level == "HIGH",
                )
                .scalar()
            )
            counts["prior_high_risk_submissions"] = prior_high
            if prior_high > 0:
                flags.append("id_previously_flagged_high_risk")

    except Exception as e:
        log.error(f"check_velocity failed: {e}")
        flags.append("velocity_check_error")
    finally:
        db.close()

    return {"flags": flags, "counts": counts}


def get_submission_history(id_number: str, limit: int = 20) -> list[dict]:
    """Return recent screening history for a given ID number."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ScreeningLog)
            .filter(ScreeningLog.id_number == id_number)
            .order_by(ScreeningLog.screened_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "screened_at": r.screened_at.isoformat(),
                "risk_level":  r.risk_level,
                "risk_score":  r.risk_score,
                "flags":       r.flags,
                "file_name":   r.file_name,
            }
            for r in rows
        ]
    finally:
        db.close()
