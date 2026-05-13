"""
auth_middleware.py — Drop-in auth check for all portals
Usage: add to top of any portal:
    from auth_middleware import require_auth, get_current_user
    Then wrap routes with: user = require_auth(request, "/coho/")
"""
import os
from typing import Optional
from datetime import datetime

import jwt
from fastapi import Request
from fastapi.responses import RedirectResponse

SECRET_KEY = os.getenv("JWT_SECRET", "")
ALGORITHM  = "HS256"

ROLE_ACCESS = {
    "admin":        ["/coho/", "/cbc/", "/trainer/", "/portal/", "/loan/", "/auth/admin"],
    "staff":        ["/coho/", "/cbc/", "/portal/"],
    "loan_officer": ["/portal/", "/loan/"],
}

def _get_token(request: Request) -> Optional[str]:
    token = request.cookies.get("auth_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    return token

def get_current_user(request: Request) -> Optional[dict]:
    """Returns user payload dict or None if not authenticated."""
    token = _get_token(request)
    if not token or not SECRET_KEY:
        return None
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return None

def require_auth(request: Request, portal_path: str = "/"):
    """
    Call at start of any route that needs auth.
    Returns user dict if authenticated and authorized.
    Raises redirect to login if not.
    """
    # Skip auth if JWT_SECRET not configured (dev mode)
    if not SECRET_KEY:
        return {"sub": "dev", "role": "admin"}

    user = get_current_user(request)
    if not user:
        raise _redirect_to_login(request)

    # Check role access
    role    = user.get("role", "")
    allowed = ROLE_ACCESS.get(role, [])
    if not any(portal_path.startswith(p) for p in allowed):
        raise _redirect_to_login(request, forbidden=True)

    return user

def _redirect_to_login(request: Request, forbidden: bool = False):
    """Return a redirect response to login page."""
    from fastapi import HTTPException
    dest = f"/auth/login?redirect={request.url.path}"
    return HTTPException(status_code=302, headers={"Location": dest})

def auth_nav_html(user: Optional[dict]) -> str:
    """Returns HTML for nav bar user info + logout button."""
    if not user:
        return ""
    role  = user.get("role", "")
    uname = user.get("sub", "")
    color = {"admin": "#dc2626", "staff": "#059669", "loan_officer": "#2563eb"}.get(role, "#64748b")
    return (
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<span style="font-size:12px;color:#94a3b8">{uname}</span>'
        f'<span style="font-size:11px;padding:2px 8px;border-radius:10px;'
        f'background:rgba(255,255,255,.1);color:{color}">{role.replace("_"," ").title()}</span>'
        f'<a href="/auth/logout" style="font-size:12px;color:#94a3b8;text-decoration:none;'
        f'padding:4px 10px;border:1px solid rgba(255,255,255,.2);border-radius:6px">Logout</a>'
        f'</div>'
    )
