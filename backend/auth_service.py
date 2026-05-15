"""
auth_service.py — Central Authentication & Authorization Service
Port: 8010  Access: /auth/

Handles:
- Login / logout
- JWT token issuance and validation
- Role-based access control
- Audit logging
- User management (admin only)
"""
import os, json, hashlib, secrets, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import bcrypt
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("auth_service")

SECRET_KEY  = os.getenv("JWT_SECRET", secrets.token_hex(32))
ALGORITHM   = "HS256"
TOKEN_EXPIRY = 8  # hours
DATA_DIR    = Path(os.getenv("UPLOAD_DIR", "/app/uploads")) / "auth"
USERS_FILE  = DATA_DIR / "users.json"
AUDIT_FILE  = DATA_DIR / "audit.log"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Auth Service")

# ── Role → allowed portals ────────────────────────────────────────────────────
ROLE_ACCESS = {
    "admin":        ["/coho/", "/cbc/", "/trainer/", "/portal/", "/loan/", "/auth/admin"],
    "staff":        ["/coho/", "/cbc/", "/portal/"],
    "loan_officer": ["/portal/", "/loan/"],
}

# ── User store ────────────────────────────────────────────────────────────────

def _load_users() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except: pass
    return {}

def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))

def _hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _check_pw(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except: return False

def _ensure_default_admin():
    """Create default admin if no users exist."""
    users = _load_users()
    if not users:
        users["admin"] = {
            "username":   "admin",
            "password":   _hash_pw("Admin@2025"),
            "role":       "admin",
            "full_name":  "System Administrator",
            "email":      "admin@bank.com",
            "active":     True,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "login_attempts": 0,
            "locked_until": None,
        }
        _save_users(users)
        log.info("Default admin created: admin / Admin@2025")

_ensure_default_admin()

# ── Audit log ─────────────────────────────────────────────────────────────────

def _audit(username: str, action: str, detail: str, ip: str = ""):
    entry = {
        "ts":       datetime.now().isoformat(),
        "user":     username,
        "action":   action,
        "detail":   detail,
        "ip":       ip,
    }
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── JWT ───────────────────────────────────────────────────────────────────────

def _create_token(username: str, role: str) -> str:
    payload = {
        "sub":  username,
        "role": role,
        "exp":  datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY),
        "iat":  datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _verify_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except: return None

def _get_token_from_request(request: Request) -> Optional[str]:
    """Get token from cookie or Authorization header."""
    token = request.cookies.get("auth_token")
    if not token:
        auth = request.headers.get("Authorization","")
        if auth.startswith("Bearer "):
            token = auth[7:]
    return token

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
:root{--nav:#1a2744;--accent:#2563eb;--ok:#059669;--danger:#dc2626;
  --surface:#f8fafc;--card:#fff;--border:#e2e8f0;--text:#0f172a;--muted:#64748b}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);min-height:100vh;
     display:flex;align-items:center;justify-content:center}
.login-box{background:var(--card);border:1px solid var(--border);border-radius:16px;
           padding:40px;width:100%;max-width:420px;box-shadow:0 4px 24px rgba(0,0,0,.08)}
.logo{text-align:center;margin-bottom:32px}
.logo-icon{width:56px;height:56px;background:#1a2744;border-radius:14px;
           display:inline-flex;align-items:center;justify-content:center;
           font-size:11px;font-weight:800;color:#fff;margin-bottom:12px}
.logo-title{font-size:20px;font-weight:700;color:#0f172a}
.logo-sub{font-size:13px;color:#64748b;margin-top:4px}
.form-group{margin-bottom:16px}
label{display:block;font-size:12px;font-weight:600;color:#475569;margin-bottom:6px}
input{width:100%;padding:10px 14px;border:1px solid var(--border);border-radius:8px;
      font-size:14px;color:var(--text);transition:border .15s}
input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.1)}
.btn{width:100%;padding:12px;background:#1a2744;color:#fff;border:none;
     border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;
     transition:background .15s;margin-top:8px}
.btn:hover{background:#163a5c}
.alert{padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:16px}
.alert-err{background:#fef2f2;border:1px solid #fca5a5;color:#991b1b}
.alert-ok{background:#ecfdf5;border:1px solid #6ee7b7;color:#065f46}
.footer{text-align:center;margin-top:24px;font-size:12px;color:#94a3b8}

/* Admin panel */
.topbar{background:var(--nav);color:#fff;height:56px;display:flex;align-items:center;
        padding:0 24px;justify-content:space-between;position:fixed;top:0;width:100%;z-index:100}
.content{max-width:1100px;margin:80px auto 40px;padding:0 20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:20px 24px;margin-bottom:18px}
.card-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;
            letter-spacing:.6px;margin-bottom:14px;padding-bottom:8px;
            border-bottom:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#D6E4F0;color:#1F4E79;padding:10px;text-align:left;font-weight:700}
td{padding:9px 10px;border-bottom:1px solid #f1f5f9}
.badge{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.badge-admin{background:#fef3c7;color:#92400e}
.badge-staff{background:#d1fae5;color:#065f46}
.badge-loan{background:#dbeafe;color:#1e40af}
.badge-active{background:#d1fae5;color:#065f46}
.badge-locked{background:#fee2e2;color:#991b1b}
.abtn{display:inline-flex;align-items:center;gap:4px;padding:5px 12px;border-radius:6px;
      font-size:12px;font-weight:500;cursor:pointer;border:none;text-decoration:none}
.abtn-primary{background:#1F4E79;color:#fff}
.abtn-danger{background:#dc2626;color:#fff}
.abtn-ghost{background:transparent;border:1px solid var(--border);color:var(--muted)}
body.admin-body{display:block;background:var(--surface)}
"""

def login_page(error="", redirect=""):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KYC System — Login</title>
<style>{CSS}</style></head>
<body>
<div class="login-box">
  <div class="logo">
    <div class="logo-icon">KYC</div>
    <div class="logo-title">Banking KYC System</div>
    <div class="logo-sub">Fraud Detection &amp; Analysis Platform</div>
  </div>
  {"<div class='alert alert-err'>" + error + "</div>" if error else ""}
  <form method="POST" action="/auth/login">
    <input type="hidden" name="redirect" value="{redirect}">
    <div class="form-group">
      <label>Username</label>
      <input type="text" name="username" placeholder="Enter username" required autofocus>
    </div>
    <div class="form-group">
      <label>Password</label>
      <input type="password" name="password" placeholder="Enter password" required>
    </div>
    <button type="submit" class="btn">Sign In</button>
  </form>
  <div class="footer">Banking KYC &amp; Fraud Detection System</div>
</div>
</body></html>"""

# ── LOGIN ─────────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
def get_login(request: Request, redirect: str = ""):
    # Already logged in with valid token?
    token = _get_token_from_request(request)
    if token:
        payload = _verify_token(token)
        if payload:
            role    = payload.get("role","")
            allowed = ROLE_ACCESS.get(role,[])
            dest    = redirect if redirect and not redirect.startswith("/auth") else (allowed[0] if allowed else "/")
            return RedirectResponse(dest, status_code=302)
    return HTMLResponse(login_page(redirect=redirect))

@app.post("/login")
async def post_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    redirect:  str = Form("/"),
):
    ip    = request.client.host
    users = _load_users()
    user  = users.get(username.lower())

    if not user:
        _audit("?", "LOGIN_FAIL", f"Unknown user: {username}", ip)
        return HTMLResponse(login_page("Invalid username or password", redirect))

    # Check lockout
    if user.get("locked_until"):
        locked = datetime.fromisoformat(user["locked_until"])
        if datetime.now() < locked:
            mins = int((locked - datetime.now()).seconds / 60) + 1
            return HTMLResponse(login_page(f"Account locked. Try again in {mins} minutes.", redirect))
        else:
            user["login_attempts"] = 0
            user["locked_until"]   = None

    if not user.get("active", True):
        _audit(username, "LOGIN_FAIL", "Account disabled", ip)
        return HTMLResponse(login_page("Account is disabled. Contact administrator.", redirect))

    if not _check_pw(password, user["password"]):
        user["login_attempts"] = user.get("login_attempts", 0) + 1
        if user["login_attempts"] >= 5:
            user["locked_until"] = (datetime.now() + timedelta(minutes=30)).isoformat()
            _audit(username, "LOCKED", "Too many failed attempts", ip)
            _save_users(users)
            return HTMLResponse(login_page("Too many failed attempts. Account locked for 30 minutes.", redirect))
        _save_users(users)
        _audit(username, "LOGIN_FAIL", f"Wrong password (attempt {user['login_attempts']})", ip)
        remaining = 5 - user["login_attempts"]
        return HTMLResponse(login_page(f"Invalid username or password. {remaining} attempts remaining.", redirect))

    # Success
    user["login_attempts"] = 0
    user["locked_until"]   = None
    user["last_login"]     = datetime.now().isoformat()
    _save_users(users)

    token = _create_token(username.lower(), user["role"])
    _audit(username, "LOGIN", f"Successful login from {ip}", ip)

    # Determine redirect
    role    = user["role"]
    allowed = ROLE_ACCESS.get(role, [])
    dest    = redirect if redirect and redirect != "/" else allowed[0] if allowed else "/"

    response = RedirectResponse(dest, status_code=302)
    response.set_cookie(
        "auth_token", token,
        httponly=True, samesite="lax",
        max_age=TOKEN_EXPIRY * 3600,
    )
    return response

@app.get("/logout")
def logout(request: Request):
    username = "unknown"
    token    = _get_token_from_request(request)
    if token:
        payload = _verify_token(token)
        if payload:
            username = payload.get("sub","?")
    _audit(username, "LOGOUT", "", request.client.host)
    response = RedirectResponse("/auth/login", status_code=302)
    response.delete_cookie("auth_token")
    return response

# ── TOKEN VALIDATION (called by other portals) ────────────────────────────────

@app.get("/validate")
def validate(request: Request, path: str = "/"):
    """
    Called by nginx auth_request to validate token.
    Returns 200 with user info headers if valid, 401 if not.
    """
    token = _get_token_from_request(request)
    if not token:
        return JSONResponse({"error": "no token"}, status_code=401)

    payload = _verify_token(token)
    if not payload:
        return JSONResponse({"error": "invalid token"}, status_code=401)

    role    = payload.get("role","")
    allowed = ROLE_ACCESS.get(role, [])

    # Check if this role can access the requested path
    path_allowed = any(path.startswith(p) for p in allowed)
    if not path_allowed:
        return JSONResponse({"error": "forbidden"}, status_code=403)

    return JSONResponse({
        "username": payload.get("sub"),
        "role":     role,
        "exp":      payload.get("exp"),
    })

# ── API: CHECK (for portals to call internally) ───────────────────────────────

@app.get("/check")
def check_auth(request: Request):
    """Used by portals to get current user info."""
    token = _get_token_from_request(request)
    if not token:
        return JSONResponse({"authenticated": False})
    payload = _verify_token(token)
    if not payload:
        return JSONResponse({"authenticated": False})
    return JSONResponse({
        "authenticated": True,
        "username":      payload.get("sub"),
        "role":          payload.get("role"),
    })

# ── ADMIN — USER MANAGEMENT ───────────────────────────────────────────────────

def _require_admin(request: Request):
    token = _get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=302, headers={"Location": "/auth/login?redirect=/auth/admin"})
    payload = _verify_token(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return payload

def admin_page(content, username=""):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>User Management</title>
<style>{CSS}</style></head>
<body class="admin-body">
<div class="topbar">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:32px;height:32px;background:#dc2626;border-radius:7px;
                display:flex;align-items:center;justify-content:center;
                font-size:10px;font-weight:800;color:#fff">KYC</div>
    <div style="font-size:14px;font-weight:700;color:#f1f5f9">User Management</div>
  </div>
  <div style="display:flex;gap:12px;align-items:center">
    <span style="color:#94a3b8;font-size:13px">{username}</span>
    <a href="/auth/logout" style="color:#94a3b8;font-size:12px;text-decoration:none">Logout</a>
  </div>
</div>
<div class="content">{content}</div>
</body></html>"""

@app.get("/admin", response_class=HTMLResponse)
def admin_users(request: Request):
    try:
        payload = _require_admin(request)
    except HTTPException as e:
        if e.status_code == 302:
            return RedirectResponse(e.headers["Location"])
        return HTMLResponse("Forbidden", status_code=403)

    users = _load_users()
    rows  = ""
    for u in users.values():
        locked   = u.get("locked_until") and datetime.fromisoformat(u["locked_until"]) > datetime.now()
        status   = '<span class="badge badge-locked">Locked</span>' if locked else \
                   ('<span class="badge badge-active">Active</span>' if u.get("active",True) else
                    '<span class="badge badge-locked">Disabled</span>')
        role_cls = {"admin":"badge-admin","staff":"badge-staff","loan_officer":"badge-loan"}.get(u["role"],"")
        last_login = u.get("last_login","Never")
        if last_login and last_login != "Never":
            last_login = last_login[:16].replace("T"," ")
        rows += f"""<tr>
          <td><strong>{u["username"]}</strong></td>
          <td>{u.get("full_name","")}</td>
          <td>{u.get("email","")}</td>
          <td><span class="badge {role_cls}">{u["role"].replace("_"," ").title()}</span></td>
          <td>{status}</td>
          <td style="font-size:12px;color:#64748b">{last_login}</td>
          <td>
            <a href="/auth/admin/edit/{u['username']}" class="abtn abtn-ghost">Edit</a>
            {"" if u["username"]=="admin" else
             f'<a href="/auth/admin/toggle/{u["username"]}" class="abtn abtn-ghost" style="margin-left:4px">'
             f'{"Enable" if not u.get("active",True) else "Disable"}</a>'
             f'<a href="/auth/admin/reset/{u["username"]}" class="abtn abtn-ghost" style="margin-left:4px">Reset PW</a>'}
          </td>
        </tr>"""

    content = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <h1 style="font-size:20px;font-weight:700">Users ({len(users)})</h1>
      <div style="display:flex;gap:10px">
        <a href="/auth/admin/audit" class="abtn abtn-ghost">Audit Log</a>
        <a href="/auth/admin/add" class="abtn abtn-primary">+ Add User</a>
      </div>
    </div>
    <div class="card">
      <table>
        <thead><tr>
          <th>Username</th><th>Full Name</th><th>Email</th>
          <th>Role</th><th>Status</th><th>Last Login</th><th>Actions</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""
    return HTMLResponse(admin_page(content, payload.get("sub","")))


@app.get("/admin/add", response_class=HTMLResponse)
def add_user_form(request: Request, msg: str = ""):
    try:
        payload = _require_admin(request)
    except HTTPException as e:
        return RedirectResponse(e.headers.get("Location","/auth/login"))

    alert = f'<div class="alert alert-ok">{msg}</div>' if msg else ""
    form  = f"""
    {alert}
    <div class="card" style="max-width:480px">
      <div class="card-title">Add New User</div>
      <form method="POST" action="/auth/admin/add">
        <div class="form-group"><label>Username *</label>
          <input name="username" required placeholder="e.g. john.doe"></div>
        <div class="form-group"><label>Full Name</label>
          <input name="full_name" placeholder="e.g. John Doe"></div>
        <div class="form-group"><label>Email</label>
          <input name="email" type="email" placeholder="john@bank.com"></div>
        <div class="form-group"><label>Role *</label>
          <select name="role" style="width:100%;padding:10px;border:1px solid var(--border);border-radius:8px">
            <option value="staff">Staff (COHO + CBC + Portal)</option>
            <option value="loan_officer">Loan Officer (Portal + Loan)</option>
            <option value="admin">Admin (All portals)</option>
          </select></div>
        <div class="form-group"><label>Password *</label>
          <input name="password" type="password" required placeholder="Min 8 characters"></div>
        <div style="display:flex;gap:10px;margin-top:16px">
          <button type="submit" class="abtn abtn-primary" style="padding:10px 20px">Add User</button>
          <a href="/auth/admin" class="abtn abtn-ghost" style="padding:10px 20px">Cancel</a>
        </div>
      </form>
    </div>"""
    return HTMLResponse(admin_page(form, payload.get("sub","")))


@app.post("/admin/add")
async def add_user(
    request:   Request,
    username:  str = Form(...),
    full_name: str = Form(""),
    email:     str = Form(""),
    role:      str = Form("staff"),
    password:  str = Form(...),
):
    try:
        payload = _require_admin(request)
    except HTTPException as e:
        return RedirectResponse(e.headers.get("Location","/auth/login"))

    if len(password) < 8:
        return RedirectResponse("/auth/admin/add?msg=Password+must+be+at+least+8+characters", status_code=302)

    users = _load_users()
    uname = username.lower().strip()
    if uname in users:
        return RedirectResponse(f"/auth/admin/add?msg=Username+{uname}+already+exists", status_code=302)

    users[uname] = {
        "username":       uname,
        "password":       _hash_pw(password),
        "role":           role,
        "full_name":      full_name,
        "email":          email,
        "active":         True,
        "created_at":     datetime.now().isoformat(),
        "last_login":     None,
        "login_attempts": 0,
        "locked_until":   None,
    }
    _save_users(users)
    _audit(payload.get("sub","admin"), "USER_CREATE", f"Created user {uname} with role {role}", request.client.host)
    return RedirectResponse(f"/auth/admin?msg=User+{uname}+created", status_code=302)


@app.get("/admin/toggle/{username}")
def toggle_user(username: str, request: Request):
    try:
        payload = _require_admin(request)
    except HTTPException as e:
        return RedirectResponse(e.headers.get("Location","/auth/login"))

    users = _load_users()
    if username in users and username != "admin":
        users[username]["active"] = not users[username].get("active", True)
        _save_users(users)
        action = "ENABLE" if users[username]["active"] else "DISABLE"
        _audit(payload.get("sub",""), f"USER_{action}", username, request.client.host)
    return RedirectResponse("/auth/admin", status_code=302)


@app.get("/admin/reset/{username}", response_class=HTMLResponse)
def reset_pw_form(username: str, request: Request):
    try:
        payload = _require_admin(request)
    except HTTPException as e:
        return RedirectResponse(e.headers.get("Location","/auth/login"))

    form = f"""
    <div class="card" style="max-width:400px">
      <div class="card-title">Reset Password — {username}</div>
      <form method="POST" action="/auth/admin/reset/{username}">
        <div class="form-group"><label>New Password *</label>
          <input name="password" type="password" required placeholder="Min 8 characters"></div>
        <div style="display:flex;gap:10px;margin-top:16px">
          <button type="submit" class="abtn abtn-primary" style="padding:10px 20px">Reset</button>
          <a href="/auth/admin" class="abtn abtn-ghost" style="padding:10px 20px">Cancel</a>
        </div>
      </form>
    </div>"""
    return HTMLResponse(admin_page(form, payload.get("sub","")))


@app.post("/admin/reset/{username}")
async def reset_pw(username: str, request: Request, password: str = Form(...)):
    try:
        payload = _require_admin(request)
    except HTTPException as e:
        return RedirectResponse(e.headers.get("Location","/auth/login"))

    if len(password) < 8:
        return RedirectResponse(f"/auth/admin/reset/{username}", status_code=302)

    users = _load_users()
    if username in users:
        users[username]["password"]       = _hash_pw(password)
        users[username]["login_attempts"] = 0
        users[username]["locked_until"]   = None
        _save_users(users)
        _audit(payload.get("sub",""), "PW_RESET", username, request.client.host)
    return RedirectResponse("/auth/admin", status_code=302)


@app.get("/admin/audit", response_class=HTMLResponse)
def audit_log(request: Request, limit: int = 100):
    try:
        payload = _require_admin(request)
    except HTTPException as e:
        return RedirectResponse(e.headers.get("Location","/auth/login"))

    rows = ""
    if AUDIT_FILE.exists():
        lines = AUDIT_FILE.read_text().strip().split("\n")
        for line in reversed(lines[-limit:]):
            try:
                e = json.loads(line)
                ts     = e.get("ts","")[:16].replace("T"," ")
                action = e.get("action","")
                color  = "#dc2626" if "FAIL" in action or "LOCK" in action else \
                         "#059669" if action == "LOGIN" else "#64748b"
                rows += f"""<tr>
                  <td style="font-size:12px;color:#64748b">{ts}</td>
                  <td><strong>{e.get("user","")}</strong></td>
                  <td><span style="color:{color};font-weight:600">{action}</span></td>
                  <td style="font-size:12px">{e.get("detail","")}</td>
                  <td style="font-size:12px;color:#94a3b8">{e.get("ip","")}</td>
                </tr>"""
            except: pass

    content = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <h1 style="font-size:20px;font-weight:700">Audit Log (last {limit})</h1>
      <a href="/auth/admin" class="abtn abtn-ghost">Back to Users</a>
    </div>
    <div class="card">
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>Time</th><th>User</th><th>Action</th><th>Detail</th><th>IP</th></tr></thead>
          <tbody>{rows or "<tr><td colspan='5' style='text-align:center;padding:20px;color:#94a3b8'>No audit entries yet</td></tr>"}</tbody>
        </table>
      </div>
    </div>"""
    return HTMLResponse(admin_page(content, payload.get("sub","")))


@app.get("/health")
def health():
    return {"status": "ok", "service": "auth"}
