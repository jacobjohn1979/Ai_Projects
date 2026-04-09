"""
portal.py — Internal Bank Staff Portal v2
Modern UI with proper navigation, user management, logout, and all features.
"""
import os, json, logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response as FResponse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("fraud_detect.portal")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI(title="Staff Portal", version="2.0.0")

RC = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}
AC = {"REJECT": "#dc2626", "REVIEW": "#d97706", "PASS": "#16a34a"}


# ── DB helper ─────────────────────────────────────────────────────────────────
def _q(sql, params={}):
    db = SessionLocal()
    try:
        return [dict(r._mapping) for r in db.execute(text(sql), params)]
    except Exception as e:
        log.error(f"Query: {e}"); return []
    finally:
        db.close()


# ── Auth helpers ──────────────────────────────────────────────────────────────
from auth import (init_auth_tables, authenticate_user, get_current_user,
                  make_session_response, can, get_all_users,
                  create_user, update_user, change_password)

ROLE_COLOR = {"admin": "#7c3aed", "reviewer": "#2563eb", "viewer": "#475569"}
ROLE_BG    = {"admin": "#f5f3ff", "reviewer": "#eff6ff", "viewer": "#f8fafc"}

def _user_initials(name):
    parts = (name or "U").split()
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else parts[0][-1])).upper()


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
:root{--nav:#1e1b4b;--nav-hover:#312e81;--nav-active:#4338ca;--accent:#4f46e5;
      --accent-h:#4338ca;--border:#e5e7eb;--surface:#f9fafb;--card:#fff;
      --text:#111827;--muted:#6b7280;--danger:#dc2626;--warn:#d97706;--ok:#16a34a}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);display:flex;min-height:100vh}
a{text-decoration:none;color:inherit}

/* Sidebar */
.sidebar{width:220px;background:var(--nav);display:flex;flex-direction:column;
          position:fixed;top:0;left:0;height:100vh;z-index:100}
.sidebar-logo{padding:20px 16px 12px;border-bottom:1px solid rgba(255,255,255,.1)}
.sidebar-logo .title{color:#fff;font-size:15px;font-weight:700;letter-spacing:.3px}
.sidebar-logo .sub{color:#a5b4fc;font-size:11px;margin-top:2px}
.nav-section{padding:12px 8px 4px;color:#818cf8;font-size:10px;font-weight:600;
              letter-spacing:.8px;text-transform:uppercase}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 16px;
          border-radius:6px;margin:1px 8px;color:#c7d2fe;font-size:13px;
          font-weight:500;cursor:pointer;transition:all .15s}
.nav-item:hover{background:var(--nav-hover);color:#fff}
.nav-item.active{background:var(--nav-active);color:#fff}
.nav-item .icon{width:16px;height:16px;opacity:.8;flex-shrink:0}
.nav-item.active .icon{opacity:1}
.sidebar-user{margin-top:auto;padding:12px 16px;border-top:1px solid rgba(255,255,255,.1)}
.user-avatar{width:32px;height:32px;border-radius:50%;background:var(--nav-active);
              display:flex;align-items:center;justify-content:center;
              color:#fff;font-size:12px;font-weight:700;flex-shrink:0}
.user-info .name{color:#e0e7ff;font-size:12px;font-weight:600}
.user-info .role{color:#818cf8;font-size:11px;text-transform:capitalize}

/* Main */
.main{margin-left:220px;flex:1;display:flex;flex-direction:column;min-height:100vh}
.topbar{background:var(--card);border-bottom:1px solid var(--border);
        padding:0 24px;height:56px;display:flex;align-items:center;
        justify-content:space-between;position:sticky;top:0;z-index:50}
.topbar-title{font-size:15px;font-weight:600;color:var(--text)}
.topbar-actions{display:flex;align-items:center;gap:8px}
.content{padding:24px;flex:1}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;
      padding:20px;margin-bottom:16px}
.card-title{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;
             letter-spacing:.5px;margin-bottom:14px}

/* Stats */
.stats{display:grid;gap:12px;margin-bottom:20px}
.stats-4{grid-template-columns:repeat(4,1fr)}
.stats-3{grid-template-columns:repeat(3,1fr)}
.stats-6{grid-template-columns:repeat(6,1fr)}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
           padding:16px 18px}
.stat-num{font-size:24px;font-weight:700;margin-bottom:2px}
.stat-lbl{font-size:11px;color:var(--muted)}

/* Table */
table{width:100%;border-collapse:collapse;font-size:13px}
th{padding:9px 12px;text-align:left;font-size:10px;font-weight:600;color:var(--muted);
   text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border);
   background:var(--surface)}
td{padding:10px 12px;border-bottom:1px solid #f3f4f6;vertical-align:middle}
tr:hover td{background:#fafafa}

/* Forms */
.form-group{margin-bottom:14px}
label.lbl{font-size:12px;font-weight:500;color:#374151;display:block;margin-bottom:5px}
input[type=text],input[type=password],input[type=email],input[type=file],
select,textarea{width:100%;padding:8px 11px;border:1px solid var(--border);
  border-radius:6px;font-size:13px;outline:none;font-family:inherit;
  background:var(--card);color:var(--text);transition:border .15s}
input:focus,select:focus,textarea:focus{border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(79,70,229,.08)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.grid-4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:6px;
     font-size:13px;font-weight:500;cursor:pointer;border:none;transition:all .15s}
.btn-primary{background:var(--accent);color:#fff}.btn-primary:hover{background:var(--accent-h)}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface);color:var(--text)}
.btn-danger{background:#fef2f2;color:var(--danger);border:1px solid #fecaca}
.btn-danger:hover{background:#fee2e2}
.btn-success{background:#f0fdf4;color:var(--ok);border:1px solid #bbf7d0}
.btn-success:hover{background:#dcfce7}
.btn-sm{padding:5px 10px;font-size:12px}

/* Badges */
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:4px;
       font-size:11px;font-weight:600}
.badge-high{background:#fef2f2;color:#991b1b}
.badge-medium{background:#fffbeb;color:#92400e}
.badge-low{background:#f0fdf4;color:#166534}
.badge-role{border-radius:4px;padding:3px 8px;font-size:11px;font-weight:600}
.tag{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;
     border-radius:3px;padding:2px 6px;font-size:11px;color:#475569;margin:2px}

/* Alerts */
.alert{padding:11px 14px;border-radius:6px;font-size:13px;margin-bottom:14px;
       display:flex;align-items:flex-start;gap:8px}
.alert-ok{background:#f0fdf4;border:1px solid #bbf7d0;color:#166534}
.alert-err{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}
.alert-warn{background:#fffbeb;border:1px solid #fde68a;color:#92400e}
.alert-info{background:#eff6ff;border:1px solid #bfdbfe;color:#1e40af}

/* Risk banner */
.risk-banner{border-radius:10px;padding:16px 20px;margin-bottom:16px;display:flex;
             align-items:center;justify-content:space-between}
.risk-high{background:#fef2f2;border:1px solid #fecaca}
.risk-medium{background:#fffbeb;border:1px solid #fde68a}
.risk-low{background:#f0fdf4;border:1px solid #bbf7d0}
.risk-level{font-size:20px;font-weight:700}
.risk-meta{font-size:13px;color:var(--muted);margin-top:3px}

/* Page header */
.page-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.page-header h1{font-size:18px;font-weight:700}

/* Toggle */
.toggle-wrap{display:flex;gap:4px;background:var(--surface);border:1px solid var(--border);
             border-radius:6px;padding:3px}
.toggle-btn{padding:5px 12px;border-radius:4px;font-size:12px;font-weight:500;
            cursor:pointer;border:none;background:transparent;color:var(--muted)}
.toggle-btn.on{background:var(--card);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.08)}

/* Empty state */
.empty{text-align:center;padding:40px 20px;color:var(--muted)}
.empty-icon{font-size:32px;margin-bottom:8px}
.empty p{font-size:13px}

/* Divider */
hr.div{border:none;border-top:1px solid var(--border);margin:16px 0}
"""


# ── SVG icons ─────────────────────────────────────────────────────────────────
ICONS = {
    "dashboard": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "submit":    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "cases":     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
    "batch":     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z"/></svg>',
    "expiry":    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "duplicate": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    "export":    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "admin":     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "logout":    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>',
    "key":       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>',
}


# ── Layout shell ──────────────────────────────────────────────────────────────
def _shell(title, content, active="", user=None, alert=""):
    role   = (user or {}).get("role", "viewer")
    uname  = (user or {}).get("username", "")
    name   = uname.title()
    init   = _user_initials(name)
    rcolor = ROLE_COLOR.get(role, "#475569")

    is_admin    = role == "admin"
    is_reviewer = role in ("admin", "reviewer")

    def nav(key, label, href, show=True):
        if not show: return ""
        cls = "nav-item active" if key == active else "nav-item"
        return f'<a href="{href}" class="{cls}">' \
               f'<span class="icon">{ICONS.get(key,"")}</span>{label}</a>'

    sidebar = f"""
    <div class="sidebar">
      <div class="sidebar-logo">
        <div class="title">KYC Portal</div>
        <div class="sub">Fraud Detection System</div>
      </div>
      <div style="overflow-y:auto;flex:1;padding:8px 0">
        <div class="nav-section">Main</div>
        {nav("dashboard","Dashboard","/portal/")}
        {nav("submit","Submit Document","/portal/submit")}
        {nav("cases","Cases","/portal/cases")}
        <div class="nav-section">Tools</div>
        {nav("batch","Batch Processing","/portal/batch")}
        {nav("expiry","Expiry Tracker","/portal/expiry")}
        {nav("duplicate","Duplicate Detection","/portal/duplicates")}
        <div class="nav-section" style="{'display:block' if is_reviewer else 'display:none'}">Reports</div>
        {nav("export","Export Data","/portal/export",is_reviewer)}
        <div class="nav-section" style="{'display:block' if is_admin else 'display:none'}">Admin</div>
        {nav("admin","User Management","/portal/admin",is_admin)}
      </div>
      <div class="sidebar-user">
        <div style="display:flex;align-items:center;gap:10px">
          <div class="user-avatar">{init}</div>
          <div class="user-info" style="flex:1;min-width:0">
            <div class="name" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{uname}</div>
            <div class="role">{role}</div>
          </div>
          <a href="/portal/logout" title="Logout" style="color:#818cf8;flex-shrink:0">
            <span style="width:16px;height:16px;display:block">{ICONS["logout"]}</span>
          </a>
        </div>
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title} — KYC Portal</title>
  <style>{CSS}</style>
</head>
<body>
  {sidebar}
  <div class="main">
    <div class="topbar">
      <div class="topbar-title">{title}</div>
      <div class="topbar-actions">
        <span style="font-size:12px;color:var(--muted)">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                background:{rcolor};margin-right:4px"></span>
          {role.title()}
        </span>
        <a href="/portal/logout" class="btn btn-ghost btn-sm">Sign out</a>
      </div>
    </div>
    <div class="content">
      {alert}
      {content}
    </div>
  </div>
</body>
</html>"""


def _alert(msg, kind="ok"):
    icons = {"ok": "✓", "err": "✕", "warn": "⚠", "info": "ℹ"}
    cls   = {"ok": "alert-ok", "err": "alert-err", "warn": "alert-warn", "info": "alert-info"}
    return f'<div class="alert {cls.get(kind,"alert-info")}"><span>{icons.get(kind,"")}</span><span>{msg}</span></div>' if msg else ""


def _badge(level):
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(level, "")
    action = {"HIGH": "REJECT", "MEDIUM": "REVIEW", "LOW": "PASS"}.get(level, level)
    return f'<span class="badge {cls}">{action}</span>'


def _risk_dot(level):
    c = RC.get(level, "#9ca3af")
    return f'<span style="display:inline-flex;align-items:center;gap:5px;font-weight:600;color:{c}"><span style="width:8px;height:8px;border-radius:50%;background:{c};flex-shrink:0"></span>{level}</span>'


# ── Auth guard ────────────────────────────────────────────────────────────────
def _guard(request, permission="view"):
    user = get_current_user(request)
    if not user:
        raise HTTPException(302, headers={"Location": "/portal/login"})
    if permission != "view" and not can(user, permission):
        raise HTTPException(403, "Insufficient permissions for this action")
    return user


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    init_auth_tables()
    try:
        from api_keys import api_keys  # ensure table exists
    except: pass


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN / LOGOUT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    err = _alert(error, "err") if error else ""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sign In — KYC Portal</title>
<style>
{CSS}
body{{background:linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);
     display:flex;align-items:center;justify-content:center;min-height:100vh}}
.login-card{{background:#fff;border-radius:14px;padding:36px 32px;width:360px;
             box-shadow:0 20px 60px rgba(0,0,0,.25)}}
.login-logo{{text-align:center;margin-bottom:24px}}
.login-logo .icon-wrap{{width:52px;height:52px;background:#eef2ff;border-radius:12px;
                        display:flex;align-items:center;justify-content:center;margin:0 auto 10px}}
.login-logo h1{{font-size:18px;font-weight:700;color:#1e1b4b}}
.login-logo p{{font-size:12px;color:#6b7280;margin-top:3px}}
.btn-login{{width:100%;padding:10px;background:#4f46e5;color:#fff;border:none;
            border-radius:7px;font-size:14px;font-weight:600;cursor:pointer;margin-top:4px}}
.btn-login:hover{{background:#4338ca}}
.divider{{text-align:center;font-size:11px;color:#9ca3af;margin:16px 0;
          position:relative}}
.divider:before{{content:'';position:absolute;top:50%;left:0;right:0;
                 border-top:1px solid #e5e7eb}}
.divider span{{background:#fff;padding:0 10px;position:relative}}
.creds{{background:#f8fafc;border-radius:6px;padding:10px 12px;font-size:11px;color:#6b7280}}
.creds code{{color:#4f46e5;font-weight:600}}
</style></head>
<body>
<div class="login-card">
  <div class="login-logo">
    <div class="icon-wrap">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" stroke-width="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
    </div>
    <h1>KYC Portal</h1>
    <p>Sign in to access the staff portal</p>
  </div>
  {err}
  <form method="post" action="/portal/login">
    <div class="form-group">
      <label class="lbl">Username</label>
      <input type="text" name="username" placeholder="Enter your username" required autofocus>
    </div>
    <div class="form-group">
      <label class="lbl">Password</label>
      <input type="password" name="password" placeholder="Enter your password" required>
    </div>
    <button type="submit" class="btn-login">Sign In</button>
  </form>
  <div class="divider"><span>default credentials</span></div>
  <div class="creds">
    Admin: <code>admin</code> / <code>admin123</code> &nbsp;·&nbsp;
    Reviewer: <code>reviewer</code> / <code>review123</code>
  </div>
</div>
</body></html>"""
    return HTMLResponse(html)


@app.post("/login")
async def login(username: str = Form(""), password: str = Form("")):
    user = authenticate_user(username, password)
    if not user:
        return RedirectResponse("/portal/login?error=Invalid+username+or+password", 303)
    return make_session_response(user["username"], user["role"], "/portal/")


@app.get("/logout")
def logout():
    r = RedirectResponse("/portal/login", 303)
    r.delete_cookie("session_token")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    user  = _guard(request)
    since = datetime.utcnow() - timedelta(days=30)
    s7    = datetime.utcnow() - timedelta(days=7)

    stats = _q("""
        SELECT COUNT(*) total,
               COUNT(*) FILTER (WHERE risk_level='HIGH')   high,
               COUNT(*) FILTER (WHERE risk_level='MEDIUM') medium,
               COUNT(*) FILTER (WHERE risk_level='LOW')    low,
               COUNT(*) FILTER (WHERE doc_type='id_card')  id_cards,
               COUNT(*) FILTER (WHERE doc_type='pdf')      pdfs,
               ROUND(AVG(risk_score),1) avg_score,
               COUNT(DISTINCT applicant_id) FILTER (WHERE applicant_id IS NOT NULL) applicants
        FROM screening_logs WHERE screened_at >= :s
    """, {"s": since})
    s = stats[0] if stats else {}

    recent = _q("""
        SELECT id, file_name, doc_type, applicant_id, risk_level, risk_score, screened_at
        FROM screening_logs ORDER BY screened_at DESC LIMIT 8
    """)

    high7 = _q("""
        SELECT id, file_name, applicant_id, risk_score, screened_at
        FROM screening_logs WHERE risk_level='HIGH' AND screened_at >= :s
        ORDER BY screened_at DESC LIMIT 5
    """, {"s": s7})

    recent_rows = ""
    for r in recent:
        level = r.get("risk_level","—")
        dt    = str(r.get("screened_at",""))[:16].replace("T"," ")
        recent_rows += (
            "<tr>"
            "<td style='font-size:12px;color:var(--muted)'>" + dt + "</td>"
            "<td style='max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
            + str(r.get("file_name",""))[:30] + "</td>"
            "<td><span class='tag'>" + str(r.get("doc_type","")).upper() + "</span></td>"
            "<td>" + str(r.get("applicant_id") or "—") + "</td>"
            "<td>" + _risk_dot(level) + "</td>"
            "<td style='text-align:center'>" + str(r.get("risk_score","")) + "</td>"
            "<td>" + _badge(level) + "</td>"
            "<td><a href='/portal/report/" + str(r.get("id")) + "' class='btn btn-ghost btn-sm'>View</a></td>"
            "</tr>"
        )

    high_rows = ""
    for r in high7:
        dt = str(r.get("screened_at",""))[:16].replace("T"," ")
        high_rows += (
            "<tr>"
            "<td style='font-size:12px;color:var(--muted)'>" + dt + "</td>"
            "<td>" + str(r.get("file_name",""))[:28] + "</td>"
            "<td>" + str(r.get("applicant_id") or "—") + "</td>"
            "<td style='font-weight:700;color:var(--danger)'>" + str(r.get("risk_score","")) + "</td>"
            "<td><a href='/portal/report/" + str(r.get("id")) + "' class='btn btn-danger btn-sm'>Review</a></td>"
            "</tr>"
        )

    high_section = (
        "<table><thead><tr><th>Date</th><th>File</th><th>Applicant</th><th>Score</th><th></th></tr></thead>"
        "<tbody>" + (high_rows or "<tr><td colspan='5' class='empty'><div class='empty-icon'>✓</div><p>No HIGH risk in last 7 days</p></td></tr>") + "</tbody></table>"
    )

    content = f"""
    <div class="stats stats-4">
      <div class="stat-card"><div class="stat-num">{s.get("total",0)}</div><div class="stat-lbl">Total Screened (30d)</div></div>
      <div class="stat-card"><div class="stat-num" style="color:var(--danger)">{s.get("high",0)}</div><div class="stat-lbl">HIGH Risk</div></div>
      <div class="stat-card"><div class="stat-num" style="color:var(--warn)">{s.get("medium",0)}</div><div class="stat-lbl">MEDIUM Risk</div></div>
      <div class="stat-card"><div class="stat-num" style="color:var(--ok)">{s.get("low",0)}</div><div class="stat-lbl">LOW Risk</div></div>
    </div>
    <div class="stats stats-4">
      <div class="stat-card"><div class="stat-num">{s.get("id_cards",0)}</div><div class="stat-lbl">ID Cards</div></div>
      <div class="stat-card"><div class="stat-num">{s.get("pdfs",0)}</div><div class="stat-lbl">PDFs</div></div>
      <div class="stat-card"><div class="stat-num">{s.get("avg_score",0)}</div><div class="stat-lbl">Avg Risk Score</div></div>
      <div class="stat-card"><div class="stat-num">{s.get("applicants",0)}</div><div class="stat-lbl">Unique Applicants</div></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title" style="color:var(--danger)">High Risk Submissions (7d)</div>
        {high_section}
      </div>
      <div class="card">
        <div class="card-title">Quick Actions</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
          <a href="/portal/submit" class="btn btn-primary" style="justify-content:center">Submit Document</a>
          <a href="/portal/batch" class="btn btn-ghost" style="justify-content:center">Batch Upload</a>
          <a href="/portal/cases" class="btn btn-ghost" style="justify-content:center">View All Cases</a>
          <a href="/portal/expiry" class="btn btn-ghost" style="justify-content:center">Expiry Tracker</a>
          <a href="/portal/duplicates" class="btn btn-ghost" style="justify-content:center">Duplicates</a>
          <a href="/portal/export" class="btn btn-ghost" style="justify-content:center">Export Data</a>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Recent Submissions</div>
      <table>
        <thead><tr><th>Date</th><th>File</th><th>Type</th><th>Applicant</th><th>Risk</th><th>Score</th><th>Action</th><th></th></tr></thead>
        <tbody>{recent_rows or "<tr><td colspan='8' class='empty'><p>No submissions yet</p></td></tr>"}</tbody>
      </table>
    </div>"""

    return HTMLResponse(_shell("Dashboard", content, "dashboard", user))


# ═══════════════════════════════════════════════════════════════════════════════
#  SUBMIT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/submit", response_class=HTMLResponse)
def submit_page(request: Request, success: str = "", error: str = ""):
    user  = _guard(request, "submit")
    alert = _alert(success, "ok") if success else (_alert(error, "err") if error else "")

    content = f"""
    {alert}
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Screen PDF Document</div>
        <form action="/portal/submit/pdf" method="post" enctype="multipart/form-data">
          <div class="form-group"><label class="lbl">Applicant ID</label>
            <input type="text" name="applicant_id" placeholder="e.g. APP-2024-001"></div>
          <div class="form-group"><label class="lbl">PDF File *</label>
            <input type="file" name="file" accept=".pdf" required></div>
          <button type="submit" class="btn btn-primary">Screen PDF</button>
        </form>
      </div>

      <div class="card">
        <div class="card-title">Screen ID Card / Passport</div>
        <form action="/portal/submit/idcard" method="post" enctype="multipart/form-data">
          <div class="form-group"><label class="lbl">Applicant ID</label>
            <input type="text" name="applicant_id" placeholder="e.g. APP-2024-001"></div>
          <div class="form-group"><label class="lbl">ID Card Image *</label>
            <input type="file" name="file" accept=".jpg,.jpeg,.png" required></div>
          <div class="form-group"><label class="lbl">Selfie (optional — enables face match)</label>
            <input type="file" name="selfie" accept=".jpg,.jpeg,.png"></div>
          <div class="form-group"><label class="lbl">Callback URL</label>
            <input type="text" name="callback_url" value="http://172.16.26.48:3000/webhook/kyc"></div>
          <button type="submit" class="btn btn-primary">Screen ID Card</button>
        </form>
      </div>

      <div class="card">
        <div class="card-title">Screen Document Image</div>
        <form action="/portal/submit/image" method="post" enctype="multipart/form-data">
          <div class="form-group"><label class="lbl">Applicant ID</label>
            <input type="text" name="applicant_id" placeholder="e.g. APP-2024-001"></div>
          <div class="form-group"><label class="lbl">Image File *</label>
            <input type="file" name="file" accept=".jpg,.jpeg,.png,.bmp,.tiff" required></div>
          <button type="submit" class="btn btn-primary">Screen Image</button>
        </form>
      </div>

      <div class="card">
        <div class="card-title">Poll ID Card Result</div>
        <div class="form-group"><label class="lbl">Task ID</label>
          <input type="text" id="tid" placeholder="Paste task_id here..."></div>
        <button class="btn btn-primary" onclick="poll()">Check Status</button>
        <div id="res" style="margin-top:12px"></div>
        <script>
        async function poll(){{
          const tid=document.getElementById('tid').value.trim();
          if(!tid)return;
          const div=document.getElementById('res');
          div.innerHTML='<div class="alert alert-info"><span>ℹ</span><span>Checking...</span></div>';
          try{{
            const r=await fetch('/result/'+tid);
            const d=await r.json();
            if(d.status==='complete'){{
              const risk=d.result?.risk||{{}};
              const c=risk.level==='HIGH'?'#dc2626':risk.level==='MEDIUM'?'#d97706':'#16a34a';
              div.innerHTML='<div class="alert alert-ok"><span>✓</span><span>Complete — Risk: <strong style="color:'+c+'">'+risk.level+'</strong> · Score: '+risk.score+' · Action: <strong>'+risk.action+'</strong></span></div>';
            }}else if(d.status==='failed'){{
              div.innerHTML='<div class="alert alert-err"><span>✕</span><span>Failed: '+d.error+'</span></div>';
            }}else{{
              div.innerHTML='<div class="alert alert-warn"><span>⚠</span><span>Status: '+d.status+' — retrying in 10s</span></div>';
              setTimeout(poll,10000);
            }}
          }}catch(e){{div.innerHTML='<div class="alert alert-err"><span>✕</span><span>Error fetching result</span></div>';}}
        }}
        </script>
      </div>
    </div>"""

    return HTMLResponse(_shell("Submit Document", content, "submit", user))


@app.post("/submit/pdf")
async def submit_pdf(request: Request, applicant_id: str = Form(""), file: UploadFile = File(...)):
    _guard(request, "submit")
    import httpx
    async with httpx.AsyncClient(base_url="http://api:8001") as c:
        r = await c.post("/screen-pdf",
            files={"file": (file.filename, await file.read(), file.content_type)},
            data={"applicant_id": applicant_id}, timeout=60)
    if r.status_code == 200:
        res  = r.json()
        risk = res.get("risk", {})
        return RedirectResponse(f"/portal/cases?msg=PDF+screened+%E2%80%94+Risk:+{risk.get('level','?')}+Score:{risk.get('score','?')}", 303)
    return RedirectResponse("/portal/submit?error=PDF+screening+failed", 303)


@app.post("/submit/image")
async def submit_image(request: Request, applicant_id: str = Form(""), file: UploadFile = File(...)):
    _guard(request, "submit")
    import httpx
    async with httpx.AsyncClient(base_url="http://api:8001") as c:
        r = await c.post("/screen-image",
            files={"file": (file.filename, await file.read(), file.content_type)},
            data={"applicant_id": applicant_id}, timeout=60)
    if r.status_code == 200:
        return RedirectResponse("/portal/cases?msg=Image+screened+successfully", 303)
    return RedirectResponse("/portal/submit?error=Image+screening+failed", 303)


@app.post("/submit/idcard")
async def submit_idcard(request: Request, applicant_id: str = Form(""), callback_url: str = Form(""),
                         file: UploadFile = File(...), selfie: UploadFile = File(None)):
    _guard(request, "submit")
    import httpx
    files = {"file": (file.filename, await file.read(), file.content_type)}
    if selfie and selfie.filename:
        files["selfie"] = (selfie.filename, await selfie.read(), selfie.content_type)
    data = {"applicant_id": applicant_id}
    if callback_url: data["callback_url"] = callback_url
    async with httpx.AsyncClient(base_url="http://api:8001") as c:
        r = await c.post("/screen-id-card", files=files, data=data, timeout=60)
    if r.status_code == 202:
        return RedirectResponse("/portal/submit?success=ID+card+queued.+Use+Poll+box+to+check+result.", 303)
    return RedirectResponse("/portal/submit?error=ID+card+submission+failed", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  CASES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/cases", response_class=HTMLResponse)
def cases_page(request: Request, applicant: str = "", risk: str = "",
               doc_type: str = "", days: int = 30, msg: str = ""):
    user  = _guard(request)
    since = datetime.utcnow() - timedelta(days=days)
    filt  = "WHERE screened_at >= :since"
    prms  = {"since": since}
    if applicant: filt += " AND applicant_id ILIKE :app"; prms["app"] = f"%{applicant}%"
    if risk:      filt += " AND risk_level = :risk"; prms["risk"] = risk
    if doc_type:  filt += " AND doc_type = :dt"; prms["dt"] = doc_type

    rows = _q(f"SELECT id, file_name, doc_type, risk_level, risk_score, flags, screened_at, applicant_id, id_number FROM screening_logs {filt} ORDER BY screened_at DESC LIMIT 100", prms)

    trs = ""
    for r in rows:
        level = r.get("risk_level","—")
        flags = r.get("flags") or []
        if isinstance(flags, str):
            try: flags = json.loads(flags)
            except: flags = []
        dt = str(r.get("screened_at",""))[:16].replace("T"," ")
        trs += (
            "<tr>"
            "<td style='font-size:12px;color:var(--muted)'>" + dt + "</td>"
            "<td style='max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>" + str(r.get("file_name",""))[:30] + "</td>"
            "<td><span class='tag'>" + str(r.get("doc_type","")).upper() + "</span></td>"
            "<td>" + str(r.get("applicant_id") or "—") + "</td>"
            "<td style='font-family:monospace;font-size:11px;color:var(--muted)'>" + str(r.get("id_number") or "—")[:12] + "</td>"
            "<td>" + _risk_dot(level) + "</td>"
            "<td style='text-align:center'>" + str(r.get("risk_score","")) + "</td>"
            "<td>" + _badge(level) + "</td>"
            "<td style='text-align:center;color:var(--muted)'>" + str(len(flags)) + "</td>"
            "<td><a href='/portal/report/" + str(r.get("id")) + "' class='btn btn-ghost btn-sm'>Report</a></td>"
            "</tr>"
        )

    # filter form
    def opt(v, label, current):
        return f'<option value="{v}" {"selected" if current==v else ""}>{label}</option>'

    filter_form = f"""
    <div class="card" style="margin-bottom:12px">
      <form method="get" style="display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap">
        <div style="flex:1;min-width:160px">
          <label class="lbl">Applicant ID</label>
          <input type="text" name="applicant" value="{applicant}" placeholder="Search...">
        </div>
        <div style="min-width:120px">
          <label class="lbl">Risk Level</label>
          <select name="risk">
            {opt("","All Risk",risk)}{opt("HIGH","HIGH",risk)}{opt("MEDIUM","MEDIUM",risk)}{opt("LOW","LOW",risk)}
          </select>
        </div>
        <div style="min-width:120px">
          <label class="lbl">Doc Type</label>
          <select name="doc_type">
            {opt("","All Types",doc_type)}{opt("id_card","ID Card",doc_type)}{opt("pdf","PDF",doc_type)}{opt("image","Image",doc_type)}
          </select>
        </div>
        <div style="min-width:100px">
          <label class="lbl">Period</label>
          <select name="days">
            {opt("7","7 days",str(days))}{opt("30","30 days",str(days))}{opt("90","90 days",str(days))}{opt("365","1 year",str(days))}
          </select>
        </div>
        <button type="submit" class="btn btn-primary">Filter</button>
        <a href="/portal/cases" class="btn btn-ghost">Reset</a>
      </form>
    </div>"""

    content = filter_form + f"""
    <div class="card">
      <div class="card-title">{len(rows)} case(s) found</div>
      <table>
        <thead><tr><th>Date</th><th>File</th><th>Type</th><th>Applicant</th><th>ID Number</th><th>Risk</th><th>Score</th><th>Action</th><th>Flags</th><th></th></tr></thead>
        <tbody>{trs or "<tr><td colspan='10' class='empty'><div class='empty-icon'>◎</div><p>No records match your filters</p></td></tr>"}</tbody>
      </table>
    </div>"""

    return HTMLResponse(_shell("Cases", content, "cases", user, _alert(msg, "ok") if msg else ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/report/{record_id}", response_class=HTMLResponse)
def view_report(record_id: int, request: Request):
    user = _guard(request)
    rows = _q("SELECT * FROM screening_logs WHERE id = :id", {"id": record_id})
    if not rows: raise HTTPException(404, "Record not found")
    r      = rows[0]
    result = r.get("full_result") or {}
    if isinstance(result, str):
        try: result = json.loads(result)
        except: result = {}

    level   = r.get("risk_level","—")
    score   = r.get("risk_score",0)
    action  = {"HIGH":"REJECT","MEDIUM":"REVIEW","LOW":"PASS"}.get(level,"—")
    flags   = r.get("flags") or []
    if isinstance(flags, str):
        try: flags = json.loads(flags)
        except: flags = []

    color   = RC.get(level,"#9ca3af")
    screened = str(r.get("screened_at",""))[:19].replace("T"," ")
    fi      = result.get("field_info",{})
    ela     = result.get("ela",{})
    holo    = result.get("hologram",{})
    tmpl    = result.get("template_match",{})
    ml      = result.get("ml_inference",{})
    face    = result.get("face_match",{})
    dec     = result.get("staff_decision",{})

    def dr(label, val, danger=False):
        style = "color:var(--danger);font-weight:600" if danger else "color:var(--text)"
        return f"<tr><td style='color:var(--muted);padding:8px 0;width:40%;font-size:13px'>{label}</td><td style='{style};padding:8px 0;font-size:13px'>{val}</td></tr>"

    mrz_ok  = fi.get("mrz_checksum_ok")
    mrz_str = ("✓ Valid" if mrz_ok is True else ("✕ Failed" if mrz_ok is False else "—"))
    flag_html = " ".join([f'<span class="tag">{f}</span>' for f in flags]) or "<span style='color:var(--muted)'>None</span>"

    banner_cls = {"HIGH":"risk-high","MEDIUM":"risk-medium","LOW":"risk-low"}.get(level,"")

    content = f"""
    <div class="page-header">
      <a href="/portal/cases" class="btn btn-ghost">← Back to Cases</a>
      <div style="display:flex;gap:8px">
        <a href="/portal/report/{record_id}/pdf" class="btn btn-success">Download PDF Report</a>
      </div>
    </div>

    <div class="risk-banner {banner_cls}">
      <div>
        <div class="risk-level" style="color:{color}">{level} RISK</div>
        <div class="risk-meta">Score: <strong>{score}</strong> · Action: <strong>{action}</strong> · {screened}</div>
      </div>
      {_badge(level)}
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Document Information</div>
        <table>
          {dr("File Name", r.get("file_name","—"))}
          {dr("Document Type", str(r.get("doc_type","—")).upper())}
          {dr("Applicant ID", r.get("applicant_id") or "—")}
          {dr("ID Number", fi.get("id_number") or "—")}
          {dr("Date of Birth", fi.get("dob") or "—")}
          {dr("Expiry Date", fi.get("expiry_date") or "—", danger="id_card_expired" in flags)}
          {dr("MRZ Checksum", mrz_str, danger=mrz_ok is False)}
          {dr("SHA-256", str(r.get("file_sha256") or "—")[:20]+"…")}
        </table>
      </div>
      <div class="card">
        <div class="card-title">Forensic Analysis</div>
        <table>
          {dr("ELA Mean Diff", ela.get("ela_mean_diff","—"))}
          {dr("ELA Std Diff", ela.get("ela_std_diff","—"))}
          {dr("Hologram", "✓ Detected" if holo.get("holographic_patch_detected") else "✕ Not detected", danger=not holo.get("holographic_patch_detected"))}
          {dr("Template Match", tmpl.get("template_matched","—"))}
          {dr("Keyword Ratio", tmpl.get("keyword_ratio","—"))}
          {dr("ML Prediction", ml.get("ml_prediction","—"))}
          {dr("ML Tamper Score", ml.get("ml_tamper_score","—"))}
          {dr("Face Match", ("✓ Verified" if face.get("face_match") else "✕ Failed") if face.get("face_match") is not None else "—", danger=face.get("face_match") is False)}
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Fraud Flags Detected ({len(flags)})</div>
      <div style="line-height:2">{flag_html}</div>
    </div>

    <div class="card">
      <div class="card-title">Staff Decision</div>
      {"<div class='alert alert-info'><span>ℹ</span><span>Decision: <strong>" + str(dec.get("decision","")) + "</strong> · " + str(dec.get("notes","")) + " · " + str(dec.get("decided_at",""))[:16] + "</span></div>" if dec else ""}
      <form method="post" action="/portal/report/{record_id}/decision">
        <div class="grid-2">
          <div class="form-group">
            <label class="lbl">Decision</label>
            <select name="decision">
              <option>PENDING REVIEW</option>
              <option>APPROVED — OVERRIDE</option>
              <option>REJECTED — CONFIRMED</option>
              <option>ESCALATED TO COMPLIANCE</option>
              <option>ADDITIONAL DOCS REQUIRED</option>
            </select>
          </div>
          <div class="form-group">
            <label class="lbl">Notes</label>
            <input type="text" name="notes" placeholder="Add notes for audit trail...">
          </div>
        </div>
        <button type="submit" class="btn btn-primary">Save Decision</button>
      </form>
    </div>"""

    return HTMLResponse(_shell(f"Report #{record_id}", content, "cases", user))


@app.post("/report/{record_id}/decision")
async def save_decision(record_id: int, request: Request,
                         decision: str = Form(""), notes: str = Form("")):
    _guard(request, "review")
    db = SessionLocal()
    try:
        db.execute(text("""
            UPDATE screening_logs
            SET full_result = jsonb_set(COALESCE(full_result,'{}')::jsonb,
                '{staff_decision}', :val::jsonb) WHERE id = :id
        """), {"id": record_id, "val": json.dumps({
            "decision": decision, "notes": notes,
            "decided_at": datetime.utcnow().isoformat()})})
        db.commit()
    except Exception as e:
        db.rollback(); log.error(f"Decision: {e}")
    finally:
        db.close()
    return RedirectResponse(f"/portal/report/{record_id}", 303)


@app.get("/report/{record_id}/pdf")
def download_pdf(record_id: int, request: Request):
    _guard(request)
    from report_gen import generate_pdf_report
    rows = _q("SELECT * FROM screening_logs WHERE id = :id", {"id": record_id})
    if not rows: raise HTTPException(404)
    pdf = generate_pdf_report(rows[0])
    if not pdf: raise HTTPException(500, "reportlab not installed")
    fname = f"KYC_Report_{record_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    return FResponse(pdf, media_type="application/pdf",
                     headers={"Content-Disposition": f"attachment; filename={fname}"})


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request, msg: str = ""):
    user = _guard(request, "submit")
    content = f"""
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Batch ID Card Screening via ZIP</div>
        <p style="font-size:13px;color:var(--muted);margin-bottom:14px">
          Upload a ZIP file of ID card images. Name each file as the applicant ID
          (e.g. <code>APP001.jpg</code>) to auto-populate applicant IDs.
        </p>
        <form action="/portal/batch/submit" method="post" enctype="multipart/form-data">
          <div class="form-group"><label class="lbl">ZIP File *</label>
            <input type="file" name="zipfile" accept=".zip" required></div>
          <div class="form-group"><label class="lbl">Callback URL</label>
            <input type="text" name="callback_url" value="http://172.16.26.48:3000/webhook/kyc"></div>
          <button type="submit" class="btn btn-primary">Start Batch</button>
        </form>
      </div>
      <div class="card">
        <div class="card-title">How Batch Processing Works</div>
        <ol style="font-size:13px;color:var(--muted);padding-left:18px;line-height:2.2">
          <li>Create a ZIP file with all ID card images</li>
          <li>Name each file as the applicant ID (APP001.jpg)</li>
          <li>Upload the ZIP and click Start Batch</li>
          <li>Each image is submitted to the screening queue</li>
          <li>Results arrive via webhook or check Cases page</li>
        </ol>
        <hr class="div">
        <a href="/portal/batch/sample.csv" class="btn btn-ghost btn-sm">Download Sample CSV</a>
      </div>
    </div>"""
    return HTMLResponse(_shell("Batch Processing", content, "batch", user, _alert(msg,"ok") if msg else ""))


@app.get("/batch/sample.csv")
def sample_csv(request: Request):
    _guard(request)
    return FResponse("applicant_id,file_path,doc_type\nAPP001,/app/test_files/id1.jpg,id_card\n",
                     media_type="text/csv",
                     headers={"Content-Disposition": "attachment; filename=sample_batch.csv"})


@app.post("/batch/submit")
async def batch_submit(request: Request, zipfile: UploadFile = File(...), callback_url: str = Form("")):
    _guard(request, "submit")
    import zipfile as zf, tempfile, httpx
    submitted = errors = 0
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "b.zip"
        zp.write_bytes(await zipfile.read())
        try:
            with zf.ZipFile(str(zp)) as z:
                names = [n for n in z.namelist() if n.lower().endswith((".jpg",".jpeg",".png"))]
                async with httpx.AsyncClient(base_url="http://api:8001") as client:
                    for name in names:
                        try:
                            data = {"applicant_id": Path(name).stem}
                            if callback_url: data["callback_url"] = callback_url
                            r = await client.post("/screen-id-card",
                                files={"file":(name,z.read(name),"image/jpeg")},
                                data=data, timeout=30)
                            submitted += 1 if r.status_code==202 else 0
                            errors    += 0 if r.status_code==202 else 1
                        except: errors += 1
        except zf.BadZipFile:
            return RedirectResponse("/portal/batch?msg=Invalid+ZIP+file", 303)
    return RedirectResponse(f"/portal/batch?msg=Batch+complete:+{submitted}+submitted,+{errors}+errors", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPIRY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/expiry", response_class=HTMLResponse)
def expiry_page(request: Request):
    user = _guard(request)
    rows = _q("""
        SELECT id, file_name, applicant_id, id_number,
               full_result->>'screened_at' AS screened_at,
               full_result->'field_info'->>'expiry_date' AS expiry_date, risk_level
        FROM screening_logs WHERE doc_type='id_card'
        AND full_result->'field_info'->>'expiry_date' IS NOT NULL
        ORDER BY screened_at DESC LIMIT 200
    """)

    now = datetime.utcnow()
    groups = {"expired":[],"exp30":[],"exp90":[],"ok":[]}
    for r in rows:
        exp = r.get("expiry_date","")
        if not exp: continue
        try:
            from dateutil import parser as dp
            dt = dp.parse(exp, dayfirst=True)
            d  = (dt-now).days
            r["days_left"] = d
            r["expiry_fmt"] = dt.strftime("%d %b %Y")
            if d < 0:      groups["expired"].append(r)
            elif d <= 30:  groups["exp30"].append(r)
            elif d <= 90:  groups["exp90"].append(r)
            else:          groups["ok"].append(r)
        except: pass

    def etable(items, color):
        if not items: return "<p style='color:var(--muted);font-size:13px;padding:8px 0'>None</p>"
        trs = ""
        for r in items:
            d = r.get("days_left",0)
            ds = (f"Expired {abs(d)}d ago" if d<0 else f"{d}d remaining")
            trs += (
                "<tr><td>" + str(r.get("applicant_id") or "—") + "</td>"
                "<td style='font-family:monospace;font-size:11px'>" + str(r.get("id_number") or "—") + "</td>"
                "<td>" + str(r.get("expiry_fmt","—")) + "</td>"
                "<td style='color:" + color + ";font-weight:600'>" + ds + "</td>"
                "<td><a href='/portal/report/" + str(r.get("id")) + "' class='btn btn-ghost btn-sm'>View</a></td>"
                "</tr>"
            )
        return "<table><thead><tr><th>Applicant</th><th>ID Number</th><th>Expiry</th><th>Status</th><th></th></tr></thead><tbody>" + trs + "</tbody></table>"

    content = f"""
    <div class="stats stats-4">
      <div class="stat-card"><div class="stat-num" style="color:var(--danger)">{len(groups["expired"])}</div><div class="stat-lbl">Expired</div></div>
      <div class="stat-card"><div class="stat-num" style="color:var(--warn)">{len(groups["exp30"])}</div><div class="stat-lbl">Expiring ≤ 30 days</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#2563eb">{len(groups["exp90"])}</div><div class="stat-lbl">Expiring ≤ 90 days</div></div>
      <div class="stat-card"><div class="stat-num" style="color:var(--ok)">{len(groups["ok"])}</div><div class="stat-lbl">Valid</div></div>
    </div>
    <div class="card" style="border-left:3px solid var(--danger)">
      <div class="card-title" style="color:var(--danger)">Expired ({len(groups["expired"])})</div>
      {etable(groups["expired"],"var(--danger)")}
    </div>
    <div class="card" style="border-left:3px solid var(--warn)">
      <div class="card-title" style="color:var(--warn)">Expiring Within 30 Days ({len(groups["exp30"])})</div>
      {etable(groups["exp30"],"var(--warn)")}
    </div>
    <div class="card" style="border-left:3px solid #2563eb">
      <div class="card-title" style="color:#2563eb">Expiring Within 90 Days ({len(groups["exp90"])})</div>
      {etable(groups["exp90"],"#2563eb")}
    </div>"""

    return HTMLResponse(_shell("Expiry Tracker", content, "expiry", user))


# ═══════════════════════════════════════════════════════════════════════════════
#  DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/duplicates", response_class=HTMLResponse)
def duplicates_page(request: Request, days: int = 90):
    user = _guard(request)
    from duplicate_detect import get_all_duplicates, get_duplicate_stats
    dupes = get_all_duplicates(days)
    stats = get_duplicate_stats(days)

    trs = ""
    for d in dupes:
        apps = ", ".join([a for a in (d.get("applicants") or []) if a])
        lvl  = str(d.get("max_risk",""))
        trs += (
            "<tr>"
            "<td style='font-family:monospace;font-size:12px'>" + str(d.get("id_number","")) + "</td>"
            "<td style='font-weight:700;color:var(--danger)'>" + str(d.get("applicant_count","")) + "</td>"
            "<td style='font-size:12px'>" + apps + "</td>"
            "<td style='text-align:center'>" + str(d.get("submission_count","")) + "</td>"
            "<td>" + _risk_dot(lvl) + "</td>"
            "<td style='font-size:12px;color:var(--muted)'>" + str(d.get("last_seen",""))[:16] + "</td>"
            "</tr>"
        )

    content = f"""
    <div class="stats stats-3" style="margin-bottom:16px">
      <div class="stat-card"><div class="stat-num" style="color:var(--danger)">{stats.get("duplicate_groups",0)}</div><div class="stat-lbl">Duplicate Groups ({days}d)</div></div>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:14px">
      <a href="/portal/duplicates?days=30" class="btn {'btn-primary' if days==30 else 'btn-ghost'} btn-sm">30 days</a>
      <a href="/portal/duplicates?days=90" class="btn {'btn-primary' if days==90 else 'btn-ghost'} btn-sm">90 days</a>
      <a href="/portal/duplicates?days=365" class="btn {'btn-primary' if days==365 else 'btn-ghost'} btn-sm">1 year</a>
    </div>
    <div class="card">
      <div class="card-title">Same ID Number, Different Applicant Names</div>
      <table>
        <thead><tr><th>ID Number</th><th>Applicant Count</th><th>Applicants</th><th>Submissions</th><th>Max Risk</th><th>Last Seen</th></tr></thead>
        <tbody>{trs or "<tr><td colspan='6' class='empty'><div class='empty-icon'>✓</div><p>No duplicate identities found</p></td></tr>"}</tbody>
      </table>
    </div>"""

    return HTMLResponse(_shell("Duplicate Detection", content, "duplicate", user))


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/export", response_class=HTMLResponse)
def export_page(request: Request):
    user = _guard(request, "export")
    content = """
    <div class="grid-3">
      <div class="card">
        <div class="card-title">Export Cases to Excel</div>
        <p style="font-size:13px;color:var(--muted);margin-bottom:14px">
          Download all screening cases with full details, colour-coded by risk level.
        </p>
        <form action="/portal/export/excel" method="get">
          <div class="form-group"><label class="lbl">Period</label>
            <select name="days"><option value="7">Last 7 days</option><option value="30" selected>Last 30 days</option><option value="90">Last 90 days</option><option value="365">Last 12 months</option></select></div>
          <div class="form-group"><label class="lbl">Risk Filter</label>
            <select name="risk"><option value="">All</option><option value="HIGH">HIGH only</option><option value="MEDIUM">MEDIUM only</option></select></div>
          <button type="submit" class="btn btn-success">Download Excel</button>
        </form>
      </div>
      <div class="card">
        <div class="card-title">Export Cases to CSV</div>
        <p style="font-size:13px;color:var(--muted);margin-bottom:14px">
          CSV format compatible with Excel, Google Sheets, and data tools.
        </p>
        <form action="/portal/export/csv" method="get">
          <div class="form-group"><label class="lbl">Period</label>
            <select name="days"><option value="7">Last 7 days</option><option value="30" selected>Last 30 days</option><option value="90">Last 90 days</option></select></div>
          <div class="form-group"><label class="lbl">Risk Filter</label>
            <select name="risk"><option value="">All</option><option value="HIGH">HIGH only</option></select></div>
          <button type="submit" class="btn btn-primary">Download CSV</button>
        </form>
      </div>
      <div class="card">
        <div class="card-title">Monthly Statistics Report</div>
        <p style="font-size:13px;color:var(--muted);margin-bottom:14px">
          12-month breakdown of all screening activity — total, risk levels, doc types.
        </p>
        <a href="/portal/export/monthly" class="btn btn-success">Download Monthly Excel</a>
      </div>
    </div>"""
    return HTMLResponse(_shell("Export Data", content, "export", user))


@app.get("/export/excel")
def export_excel(request: Request, days: int = 30, risk: str = ""):
    _guard(request, "export")
    from export import export_cases_excel
    data = export_cases_excel(days, risk)
    if not data: raise HTTPException(500, "openpyxl not installed")
    fname = f"KYC_Cases_{days}d_{datetime.utcnow().strftime('%Y%m%d')}.xlsx"
    return FResponse(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.get("/export/csv")
def export_csv(request: Request, days: int = 30, risk: str = ""):
    _guard(request, "export")
    from export import export_cases_csv
    data = export_cases_csv(days, risk)
    fname = f"KYC_Cases_{days}d_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    return FResponse(data, media_type="text/csv",
                     headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.get("/export/monthly")
def export_monthly(request: Request):
    _guard(request, "export")
    from export import export_monthly_stats_excel
    data = export_monthly_stats_excel()
    if not data: raise HTTPException(500, "openpyxl not installed")
    fname = f"KYC_Monthly_{datetime.utcnow().strftime('%Y%m')}.xlsx"
    return FResponse(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     headers={"Content-Disposition": f"attachment; filename={fname}"})


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN — USER MANAGEMENT + API KEYS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, msg: str = "", error: str = ""):
    user = _guard(request, "manage_users")
    from api_keys import get_all_keys

    users    = get_all_users()
    api_keys = get_all_keys()

    user_rows = ""
    for u in users:
        role    = u.get("role","viewer")
        rcolor  = ROLE_COLOR.get(role,"#475569")
        rbg     = ROLE_BG.get(role,"#f8fafc")
        active  = u.get("active", True)
        ll      = str(u.get("last_login",""))[:16] or "Never"
        init    = _user_initials(u.get("name") or u.get("username","?"))
        status_badge = (
            '<span style="background:#f0fdf4;color:#166534;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600">Active</span>'
            if active else
            '<span style="background:#fef2f2;color:#991b1b;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600">Inactive</span>'
        )
        user_rows += (
            "<tr>"
            "<td><div style='display:flex;align-items:center;gap:10px'>"
            "<div style='width:30px;height:30px;border-radius:50%;background:" + rcolor + ";display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700;flex-shrink:0'>" + init + "</div>"
            "<div><div style='font-weight:500;font-size:13px'>" + str(u.get("username","")) + "</div>"
            "<div style='color:var(--muted);font-size:11px'>" + str(u.get("email") or "") + "</div></div>"
            "</div></td>"
            "<td>" + str(u.get("name","")) + "</td>"
            "<td><span style='background:" + rbg + ";color:" + rcolor + ";border-radius:4px;padding:3px 8px;font-size:11px;font-weight:600'>" + role.title() + "</span></td>"
            "<td>" + status_badge + "</td>"
            "<td style='font-size:12px;color:var(--muted)'>" + ll + "</td>"
            "<td>"
            "<form method='post' action='/portal/admin/toggle-user' style='display:inline'>"
            "<input type='hidden' name='user_id' value='" + str(u.get("id","")) + "'>"
            "<input type='hidden' name='active' value='" + ("false" if active else "true") + "'>"
            "<button class='btn btn-ghost btn-sm'>" + ("Deactivate" if active else "Activate") + "</button>"
            "</form>"
            "</td>"
            "</tr>"
        )

    key_rows = ""
    for k in api_keys:
        active  = k.get("active", True)
        lu      = str(k.get("last_used",""))[:16] or "Never"
        key_rows += (
            "<tr>"
            "<td style='font-family:monospace;font-size:12px'>" + str(k.get("key_prefix","")) + "…</td>"
            "<td>" + str(k.get("name","")) + "</td>"
            "<td style='color:var(--muted);font-size:12px'>" + str(k.get("created_by","")) + "</td>"
            "<td>" + ('<span style="color:var(--ok);font-weight:600">Active</span>' if active else '<span style="color:var(--muted)">Revoked</span>') + "</td>"
            "<td style='font-size:12px;color:var(--muted)'>" + lu + "</td>"
            "<td>"
            "<form method='post' action='/portal/admin/revoke-key' style='display:inline'>"
            "<input type='hidden' name='key_id' value='" + str(k.get("id","")) + "'>"
            "<button class='btn btn-danger btn-sm' " + ("" if active else "disabled") + ">Revoke</button>"
            "</form>"
            "</td>"
            "</tr>"
        )

    alert = _alert(msg, "ok") if msg else (_alert(error, "err") if error else "")

    content = f"""
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Staff Users</div>
        <table style="margin-bottom:16px">
          <thead><tr><th>User</th><th>Name</th><th>Role</th><th>Status</th><th>Last Login</th><th></th></tr></thead>
          <tbody>{user_rows or "<tr><td colspan='6' class='empty'><p>No users found</p></td></tr>"}</tbody>
        </table>
        <hr class="div">
        <div class="card-title">Add New User</div>
        <form method="post" action="/portal/admin/create-user">
          <div class="grid-2">
            <div class="form-group"><label class="lbl">Username *</label><input name="username" required></div>
            <div class="form-group"><label class="lbl">Password *</label><input type="password" name="password" required></div>
            <div class="form-group"><label class="lbl">Full Name *</label><input name="name" required></div>
            <div class="form-group">
              <label class="lbl">Role</label>
              <select name="role">
                <option value="viewer">Viewer — view only</option>
                <option value="reviewer">Reviewer — submit + review</option>
                <option value="admin">Admin — full access</option>
              </select>
            </div>
          </div>
          <div class="form-group"><label class="lbl">Email</label><input type="email" name="email" placeholder="For alerts"></div>
          <button type="submit" class="btn btn-primary">Create User</button>
        </form>
      </div>

      <div class="card">
        <div class="card-title">API Keys</div>
        <div class="alert alert-info"><span>ℹ</span><span>Enable API auth in .env: <code>API_AUTH_ENABLED=true</code>. Use header: <code>X-API-Key: fds_your_key</code></span></div>
        <table style="margin-bottom:16px">
          <thead><tr><th>Prefix</th><th>Name</th><th>Created By</th><th>Status</th><th>Last Used</th><th></th></tr></thead>
          <tbody>{key_rows or "<tr><td colspan='6' class='empty'><p>No API keys yet</p></td></tr>"}</tbody>
        </table>
        <hr class="div">
        <div class="card-title">Generate New API Key</div>
        <div class="alert alert-warn"><span>⚠</span><span>The key is shown only once after creation — copy it immediately.</span></div>
        <form method="post" action="/portal/admin/create-key">
          <div class="form-group"><label class="lbl">Key Name / Description *</label>
            <input name="key_name" placeholder="e.g. Mobile App, Partner Integration" required></div>
          <button type="submit" class="btn btn-primary">Generate Key</button>
        </form>

        <hr class="div">
        <div class="card-title">Change Password</div>
        <form method="post" action="/portal/admin/change-password">
          <div class="grid-2">
            <div class="form-group"><label class="lbl">Username</label><input name="username" required></div>
            <div class="form-group"><label class="lbl">New Password</label><input type="password" name="new_password" required></div>
          </div>
          <button type="submit" class="btn btn-ghost">Update Password</button>
        </form>

        <hr class="div">
        <div class="card-title">Email Alert Config</div>
        <div class="alert alert-info"><span>ℹ</span><span>Configure in server .env file — SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_TO, ALERTS_ENABLED=true</span></div>
        <a href="/portal/admin/test-email" class="btn btn-ghost btn-sm">Test Email Config</a>
      </div>
    </div>"""

    return HTMLResponse(_shell("Admin — User Management", content, "admin", user, alert))


@app.post("/admin/create-user")
async def create_user_route(request: Request,
    username: str=Form(""), password: str=Form(""),
    role: str=Form("viewer"), name: str=Form(""), email: str=Form("")):
    _guard(request, "manage_users")
    ok = create_user(username, password, role, name, email)
    if ok: return RedirectResponse(f"/portal/admin?msg=User+{username}+created+successfully", 303)
    return RedirectResponse("/portal/admin?error=Failed+to+create+user+(username+may+already+exist)", 303)


@app.post("/admin/toggle-user")
async def toggle_user(request: Request, user_id: int=Form(...), active: str=Form("true")):
    _guard(request, "manage_users")
    current_user = get_current_user(request)
    users = get_all_users()
    u = next((x for x in users if x["id"]==user_id), None)
    if u and u["username"] == current_user.get("username"):
        return RedirectResponse("/portal/admin?error=Cannot+deactivate+your+own+account", 303)
    db = SessionLocal()
    try:
        db.execute(text("UPDATE staff_users SET active=:a WHERE id=:id"),
                   {"a": active=="true", "id": user_id})
        db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()
    return RedirectResponse("/portal/admin?msg=User+status+updated", 303)


@app.post("/admin/change-password")
async def change_pwd(request: Request, username: str=Form(""), new_password: str=Form("")):
    _guard(request, "manage_users")
    ok = change_password(username, new_password)
    if ok: return RedirectResponse("/portal/admin?msg=Password+changed+for+"+username, 303)
    return RedirectResponse("/portal/admin?error=Failed+to+change+password", 303)


@app.post("/admin/create-key")
async def create_key_route(request: Request, key_name: str=Form("")):
    user = _guard(request, "manage_users")
    from api_keys import create_api_key
    key = create_api_key(key_name, user["username"])
    return RedirectResponse("/portal/admin?msg=API+Key+created+(copy+now):+" + key, 303)


@app.post("/admin/revoke-key")
async def revoke_key_route(request: Request, key_id: int=Form(...)):
    _guard(request, "manage_users")
    from api_keys import revoke_api_key
    revoke_api_key(key_id)
    return RedirectResponse("/portal/admin?msg=API+key+revoked", 303)


@app.get("/admin/test-email")
def test_email(request: Request):
    _guard(request, "manage_users")
    from alerts import test_email_config
    cfg = test_email_config()
    return JSONResponse(cfg)


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "running", "service": "portal", "version": "2.0.0"}
