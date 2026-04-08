"""
api_keys.py — API Key Authentication
Secures /screen-* endpoints with rotating API keys.
"""
import os
import hashlib
import secrets
import logging
from datetime import datetime
from fastapi import Request, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("fraud_detect.apikeys")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new API key in format: fds_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"""
    return "fds_" + secrets.token_hex(24)


def create_api_key(name: str, created_by: str) -> str:
    """Create and store a new API key. Returns the plain key (shown once only)."""
    key    = generate_api_key()
    prefix = key[:10]
    db     = SessionLocal()
    try:
        db.execute(text("""
            INSERT INTO api_keys (key_hash, key_prefix, name, created_by)
            VALUES (:h, :p, :n, :c)
        """), {"h": _hash_key(key), "p": prefix, "n": name, "c": created_by})
        db.commit()
        log.info(f"API key created: {prefix}… by {created_by}")
        return key
    except Exception as e:
        db.rollback()
        log.error(f"create_api_key failed: {e}")
        raise
    finally:
        db.close()


def verify_api_key(key: str) -> bool:
    """Verify an API key. Updates last_used timestamp."""
    if not key:
        return False
    db = SessionLocal()
    try:
        row = db.execute(text("""
            SELECT id FROM api_keys
            WHERE key_hash = :h AND active = TRUE
        """), {"h": _hash_key(key)}).fetchone()
        if row:
            db.execute(text("UPDATE api_keys SET last_used=NOW() WHERE id=:id"),
                       {"id": row.id})
            db.commit()
            return True
        return False
    finally:
        db.close()


def get_all_keys() -> list:
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT id, key_prefix, name, created_by, active, created_at, last_used
            FROM api_keys ORDER BY created_at DESC
        """)).fetchall()
        result = []
        for r in rows:
            rd = dict(r._mapping)
            if rd.get("created_at"): rd["created_at"] = rd["created_at"].isoformat()
            if rd.get("last_used"):  rd["last_used"]  = rd["last_used"].isoformat()
            result.append(rd)
        return result
    finally:
        db.close()


def revoke_api_key(key_id: int) -> bool:
    db = SessionLocal()
    try:
        db.execute(text("UPDATE api_keys SET active=FALSE WHERE id=:id"), {"id": key_id})
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        return False
    finally:
        db.close()


def require_api_key(request: Request):
    """FastAPI dependency — checks API key if auth is enabled."""
    if not API_AUTH_ENABLED:
        return  # auth disabled — allow all

    # check header: Authorization: Bearer fds_xxx or X-API-Key: fds_xxx
    key = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        key = auth_header[7:]
    if not key:
        key = request.headers.get("X-API-Key", "")

    if not key or not verify_api_key(key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include: X-API-Key: your_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
