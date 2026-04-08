"""
auth.py — Staff Authentication System
JWT-based login with 3 roles: admin, reviewer, viewer
"""
import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("fraud_detect.auth")

JWT_SECRET   = os.getenv("JWT_SECRET", secrets.token_hex(32))
SESSION_HOURS = int(os.getenv("SESSION_HOURS", "8"))

# ── Role permissions ──────────────────────────────────────────────────────────
ROLE_PERMISSIONS = {
    "admin":    ["view", "submit", "review", "export", "manage_users", "manage_keys"],
    "reviewer": ["view", "submit", "review", "export"],
    "viewer":   ["view"],
}

# ── Default users (stored in db — these are just seeds) ──────────────────────
DEFAULT_USERS = [
    {"username": "admin",    "password": "admin123",    "role": "admin",    "name": "Administrator"},
    {"username": "reviewer", "password": "review123",   "role": "reviewer", "name": "Document Reviewer"},
    {"username": "viewer",   "password": "viewer123",   "role": "viewer",   "name": "Viewer"},
]


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _make_token(username: str, role: str) -> str:
    """Simple signed token: base64(payload).signature"""
    import base64
    import hmac
    payload = json.dumps({
        "username": username,
        "role":     role,
        "exp":      (datetime.utcnow() + timedelta(hours=SESSION_HOURS)).isoformat(),
    })
    encoded = base64.b64encode(payload.encode()).decode()
    sig = hmac.new(JWT_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return encoded + "." + sig


def _verify_token(token: str) -> dict | None:
    """Verify token signature and expiry. Returns payload or None."""
    import base64
    import hmac
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        encoded, sig = parts
        expected = hmac.new(JWT_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.b64decode(encoded).decode())
        exp = datetime.fromisoformat(payload["exp"])
        if datetime.utcnow() > exp:
            return None
        return payload
    except Exception:
        return None


# ── Database helpers ──────────────────────────────────────────────────────────
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def init_auth_tables():
    """Create staff_users and api_keys tables."""
    db = SessionLocal()
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS staff_users (
                id         SERIAL PRIMARY KEY,
                username   VARCHAR(50) UNIQUE NOT NULL,
                password   VARCHAR(64) NOT NULL,
                role       VARCHAR(20) NOT NULL DEFAULT 'viewer',
                name       VARCHAR(100),
                email      VARCHAR(100),
                active     BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                last_login TIMESTAMP
            )
        """))
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id          SERIAL PRIMARY KEY,
                key_hash    VARCHAR(64) UNIQUE NOT NULL,
                key_prefix  VARCHAR(10) NOT NULL,
                name        VARCHAR(100),
                created_by  VARCHAR(50),
                active      BOOLEAN DEFAULT TRUE,
                last_used   TIMESTAMP,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """))
        # seed default users if none exist
        count = db.execute(text("SELECT COUNT(*) FROM staff_users")).scalar()
        if count == 0:
            for u in DEFAULT_USERS:
                db.execute(text("""
                    INSERT INTO staff_users (username, password, role, name)
                    VALUES (:u, :p, :r, :n)
                """), {"u": u["username"], "p": _hash_password(u["password"]),
                       "r": u["role"], "n": u["name"]})
            log.info("Default staff users created")
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"init_auth_tables failed: {e}")
    finally:
        db.close()


def authenticate_user(username: str, password: str) -> dict | None:
    db = SessionLocal()
    try:
        row = db.execute(text("""
            SELECT username, role, name, email FROM staff_users
            WHERE username = :u AND password = :p AND active = TRUE
        """), {"u": username, "p": _hash_password(password)}).fetchone()
        if row:
            db.execute(text("UPDATE staff_users SET last_login = NOW() WHERE username = :u"),
                       {"u": username})
            db.commit()
            return dict(row._mapping)
        return None
    finally:
        db.close()


def get_all_users() -> list:
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT id, username, role, name, email, active, created_at, last_login FROM staff_users ORDER BY id"
        )).fetchall()
        return [dict(r._mapping) for r in rows]
    finally:
        db.close()


def create_user(username: str, password: str, role: str, name: str, email: str = "") -> bool:
    db = SessionLocal()
    try:
        db.execute(text("""
            INSERT INTO staff_users (username, password, role, name, email)
            VALUES (:u, :p, :r, :n, :e)
        """), {"u": username, "p": _hash_password(password), "r": role, "n": name, "e": email})
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        log.error(f"create_user failed: {e}")
        return False


def update_user(user_id: int, role: str, active: bool, name: str) -> bool:
    db = SessionLocal()
    try:
        db.execute(text("""
            UPDATE staff_users SET role=:r, active=:a, name=:n WHERE id=:id
        """), {"r": role, "a": active, "n": name, "id": user_id})
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        return False


def change_password(username: str, new_password: str) -> bool:
    db = SessionLocal()
    try:
        db.execute(text("UPDATE staff_users SET password=:p WHERE username=:u"),
                   {"p": _hash_password(new_password), "u": username})
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        return False


# ── Request helpers ───────────────────────────────────────────────────────────

def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get("session_token")
    if not token:
        return None
    return _verify_token(token)


def require_auth(request: Request, permission: str = "view") -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(302, headers={"Location": "/portal/login"})
    perms = ROLE_PERMISSIONS.get(user.get("role", "viewer"), [])
    if permission not in perms:
        raise HTTPException(403, "Insufficient permissions")
    return user


def can(user: dict, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(user.get("role", "viewer"), [])


def make_session_response(username: str, role: str, redirect_to: str = "/portal/"):
    token    = _make_token(username, role)
    response = RedirectResponse(redirect_to, status_code=303)
    response.set_cookie(
        "session_token", token,
        httponly = True,
        samesite = "lax",
        max_age  = SESSION_HOURS * 3600,
    )
    return response
