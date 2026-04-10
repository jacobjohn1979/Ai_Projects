"""
loan_db.py — Loan Portal Database
Separate tables from the KYC fraud detection system.
"""
import os
import hashlib
import secrets
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    DateTime, JSON, Text, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
log = logging.getLogger("loan_portal.db")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id                = Column(Integer, primary_key=True, index=True)
    loan_ref          = Column(String(50), unique=True, index=True)

    # Loan details
    loan_type         = Column(String(50))   # personal | business | mortgage
    loan_amount       = Column(Float)
    loan_purpose      = Column(String(200))
    loan_term_months  = Column(Integer)
    interest_rate     = Column(Float, nullable=True)

    # Applicant details
    applicant_id      = Column(String(50), index=True)
    applicant_name    = Column(String(200))
    applicant_dob     = Column(String(20))
    applicant_phone   = Column(String(30))
    applicant_email   = Column(String(100))
    applicant_address = Column(Text)
    applicant_employer= Column(String(200))
    applicant_income  = Column(Float, nullable=True)

    # Documents uploaded
    docs_uploaded     = Column(JSON, default=list)   # list of doc names

    # KYC screening result
    kyc_status        = Column(String(20), default="pending")  # pending|screening|complete|failed
    kyc_risk_level    = Column(String(10), nullable=True)      # LOW|MEDIUM|HIGH
    kyc_risk_score    = Column(Integer, nullable=True)
    kyc_action        = Column(String(10), nullable=True)      # PASS|REVIEW|REJECT
    kyc_flags         = Column(JSON, nullable=True)
    kyc_result        = Column(JSON, nullable=True)
    kyc_screened_at   = Column(DateTime, nullable=True)

    # Loan decision
    status            = Column(String(20), default="draft")
    # draft | submitted | screening | review | approved | rejected | cancelled
    decision          = Column(String(20), nullable=True)
    decision_notes    = Column(Text, nullable=True)
    decided_by        = Column(String(100), nullable=True)
    decided_at        = Column(DateTime, nullable=True)

    # Officer
    created_by        = Column(String(100))
    assigned_to       = Column(String(100), nullable=True)

    # Timestamps
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    submitted_at      = Column(DateTime, nullable=True)


class LoanOfficer(Base):
    __tablename__ = "loan_officers"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(50), unique=True, index=True)
    password   = Column(String(64))
    name       = Column(String(100))
    email      = Column(String(100))
    role       = Column(String(20), default="officer")  # officer | manager | admin
    active     = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class LoanComment(Base):
    __tablename__ = "loan_comments"

    id         = Column(Integer, primary_key=True, index=True)
    loan_ref   = Column(String(50), index=True)
    author     = Column(String(100))
    comment    = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Index("ix_loan_status",    LoanApplication.status)
Index("ix_loan_kyc",       LoanApplication.kyc_status)
Index("ix_loan_created",   LoanApplication.created_at)


# ── Init ──────────────────────────────────────────────────────────────────────

DEFAULT_OFFICERS = [
    {"username": "admin",   "password": "admin123",    "role": "admin",   "name": "Admin"},
    {"username": "officer", "password": "officer123",  "role": "officer", "name": "Loan Officer"},
    {"username": "manager", "password": "manager123",  "role": "manager", "name": "Branch Manager"},
]


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def init_loan_db():
    try:
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        count = db.execute(__import__("sqlalchemy").text("SELECT COUNT(*) FROM loan_officers")).scalar()
        if count == 0:
            for o in DEFAULT_OFFICERS:
                db.add(LoanOfficer(
                    username=o["username"], password=_hash(o["password"]),
                    role=o["role"], name=o["name"]
                ))
            db.commit()
            log.info("Default loan officers created")
        db.close()
    except Exception as e:
        log.error(f"loan_db init: {e}")


# ── Auth helpers ──────────────────────────────────────────────────────────────

import base64
import hmac
import json

JWT_SECRET    = os.getenv("JWT_SECRET", secrets.token_hex(32))
SESSION_HOURS = 8


def make_token(username: str, role: str) -> str:
    payload = json.dumps({"username": username, "role": role,
                          "exp": (datetime.utcnow().__class__.utcnow() if False else
                                  datetime.utcnow().isoformat())})
    encoded = base64.b64encode(payload.encode()).decode()
    sig     = hmac.new(JWT_SECRET.encode(), encoded.encode(), __import__("hashlib").sha256).hexdigest()
    return encoded + "." + sig


def verify_token(token: str) -> dict | None:
    try:
        encoded, sig = token.split(".", 1)
        expected = hmac.new(JWT_SECRET.encode(), encoded.encode(), __import__("hashlib").sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.b64decode(encoded).decode())
        return payload
    except Exception:
        return None


def authenticate(username: str, password: str) -> dict | None:
    db = SessionLocal()
    try:
        from sqlalchemy import text
        row = db.execute(text(
            "SELECT id, username, role, name FROM loan_officers WHERE username=:u AND password=:p AND active=TRUE"
        ), {"u": username, "p": _hash(password)}).fetchone()
        if row:
            db.execute(text("UPDATE loan_officers SET last_login=NOW() WHERE username=:u"), {"u": username})
            db.commit()
            return dict(row._mapping)
        return None
    finally:
        db.close()


def get_all_officers() -> list:
    db = SessionLocal()
    try:
        from sqlalchemy import text
        rows = db.execute(text(
            "SELECT id, username, name, role, email, active, created_at, last_login FROM loan_officers ORDER BY id"
        )).fetchall()
        result = []
        for r in rows:
            rd = dict(r._mapping)
            for k in ["created_at","last_login"]:
                if rd.get(k): rd[k] = rd[k].isoformat()
            result.append(rd)
        return result
    finally:
        db.close()


# ── Loan CRUD ─────────────────────────────────────────────────────────────────

def _gen_ref() -> str:
    import random, string
    return "LN" + datetime.utcnow().strftime("%y%m%d") + "".join(random.choices(string.digits, k=4))


def create_loan(data: dict) -> LoanApplication:
    db = SessionLocal()
    try:
        loan = LoanApplication(
            loan_ref          = _gen_ref(),
            loan_type         = data.get("loan_type", "personal"),
            loan_amount       = float(data.get("loan_amount", 0)),
            loan_purpose      = data.get("loan_purpose", ""),
            loan_term_months  = int(data.get("loan_term_months", 12)),
            applicant_id      = data.get("applicant_id", ""),
            applicant_name    = data.get("applicant_name", ""),
            applicant_dob     = data.get("applicant_dob", ""),
            applicant_phone   = data.get("applicant_phone", ""),
            applicant_email   = data.get("applicant_email", ""),
            applicant_address = data.get("applicant_address", ""),
            applicant_employer= data.get("applicant_employer", ""),
            applicant_income  = float(data.get("applicant_income", 0) or 0),
            status            = "draft",
            created_by        = data.get("created_by", ""),
        )
        db.add(loan)
        db.commit()
        db.refresh(loan)
        return loan
    finally:
        db.close()


def get_loan(loan_ref: str) -> dict | None:
    db = SessionLocal()
    try:
        from sqlalchemy import text
        row = db.execute(text("SELECT * FROM loan_applications WHERE loan_ref=:r"), {"r": loan_ref}).fetchone()
        if not row: return None
        r = dict(row._mapping)
        for k in ["created_at","updated_at","submitted_at","kyc_screened_at","decided_at"]:
            if r.get(k): r[k] = r[k].isoformat()
        return r
    finally:
        db.close()


def list_loans(status: str = "", officer: str = "", limit: int = 100) -> list:
    db = SessionLocal()
    try:
        from sqlalchemy import text
        sql = "SELECT loan_ref, loan_type, loan_amount, applicant_name, applicant_id, status, kyc_risk_level, kyc_action, created_by, created_at, decided_at FROM loan_applications WHERE 1=1"
        params = {}
        if status:
            sql += " AND status=:s"; params["s"] = status
        if officer:
            sql += " AND (created_by=:o OR assigned_to=:o)"; params["o"] = officer
        sql += " ORDER BY created_at DESC LIMIT :lim"; params["lim"] = limit
        rows = db.execute(text(sql), params).fetchall()
        result = []
        for r in rows:
            rd = dict(r._mapping)
            for k in ["created_at","decided_at"]:
                if rd.get(k): rd[k] = rd[k].isoformat()
            result.append(rd)
        return result
    finally:
        db.close()


def update_loan(loan_ref: str, updates: dict):
    db = SessionLocal()
    try:
        from sqlalchemy import text
        sets   = ", ".join(f"{k}=:{k}" for k in updates)
        params = {**updates, "ref": loan_ref}
        db.execute(text(f"UPDATE loan_applications SET {sets}, updated_at=NOW() WHERE loan_ref=:ref"), params)
        db.commit()
    except Exception as e:
        db.rollback(); log.error(f"update_loan: {e}")
    finally:
        db.close()


def add_comment(loan_ref: str, author: str, comment: str):
    db = SessionLocal()
    try:
        db.add(LoanComment(loan_ref=loan_ref, author=author, comment=comment))
        db.commit()
    finally:
        db.close()


def get_comments(loan_ref: str) -> list:
    db = SessionLocal()
    try:
        from sqlalchemy import text
        rows = db.execute(text(
            "SELECT author, comment, created_at FROM loan_comments WHERE loan_ref=:r ORDER BY created_at"
        ), {"r": loan_ref}).fetchall()
        return [{"author": r.author, "comment": r.comment,
                 "created_at": str(r.created_at)[:16]} for r in rows]
    finally:
        db.close()


def get_stats() -> dict:
    db = SessionLocal()
    try:
        from sqlalchemy import text
        r = db.execute(text("""
            SELECT
                COUNT(*) total,
                COUNT(*) FILTER (WHERE status='draft')      draft,
                COUNT(*) FILTER (WHERE status='screening')  screening,
                COUNT(*) FILTER (WHERE status='review')     review,
                COUNT(*) FILTER (WHERE status='approved')   approved,
                COUNT(*) FILTER (WHERE status='rejected')   rejected,
                COUNT(*) FILTER (WHERE kyc_risk_level='HIGH')   high_risk,
                COUNT(*) FILTER (WHERE kyc_risk_level='MEDIUM') medium_risk,
                ROUND(AVG(loan_amount)::numeric, 0)         avg_amount,
                SUM(loan_amount) FILTER (WHERE status='approved') approved_volume
            FROM loan_applications
            WHERE created_at >= NOW() - INTERVAL '30 days'
        """)).fetchone()
        return dict(r._mapping) if r else {}
    finally:
        db.close()
