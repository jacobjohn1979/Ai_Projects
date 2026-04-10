"""
loan_portal.py — Loan Officer Portal
Separate portal for managing loan applications with KYC integration.
Runs on port 8005, accessible at /loan/ via Nginx.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response as FResponse
from dotenv import load_dotenv

from loan_db import (
    init_loan_db, authenticate, verify_token, make_token,
    create_loan, get_loan, list_loans, update_loan,
    add_comment, get_comments, get_stats, get_all_officers,
    _hash, SessionLocal, LoanOfficer
)

load_dotenv()
log = logging.getLogger("loan_portal")

KYC_API_URL = os.getenv("KYC_API_URL", "http://api:8001")
KYC_API_KEY = os.getenv("KYC_API_KEY", "")

app = FastAPI(title="Loan Officer Portal", version="1.0.0")

# ── Colours ───────────────────────────────────────────────────────────────────
STATUS_COLOR = {
    "draft":     "#94a3b8",
    "submitted": "#3b82f6",
    "screening": "#8b5cf6",
    "review":    "#f59e0b",
    "approved":  "#22c55e",
    "rejected":  "#ef4444",
    "cancelled": "#64748b",
}
RISK_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
LOAN_TYPES = ["Personal Loan", "Business / SME Loan", "Mortgage / Home Loan",
              "Vehicle Loan", "Education Loan", "Agricultural Loan"]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
:root{--nav:#0c4a6e;--nav-hover:#0369a1;--nav-active:#0284c7;--accent:#0284c7;
      --accent-h:#0369a1;--border:#e5e7eb;--surface:#f0f9ff;--card:#fff;
      --text:#0f172a;--muted:#64748b;--danger:#dc2626;--warn:#d97706;--ok:#16a34a}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);display:flex;min-height:100vh}
a{text-decoration:none;color:inherit}
.sidebar{width:220px;background:var(--nav);display:flex;flex-direction:column;
         position:fixed;top:0;left:0;height:100vh;z-index:100}
.sidebar-logo{padding:20px 16px 14px;border-bottom:1px solid rgba(255,255,255,.1)}
.sidebar-logo .title{color:#fff;font-size:14px;font-weight:700}
.sidebar-logo .sub{color:#7dd3fc;font-size:11px;margin-top:2px}
.nav-section{padding:12px 8px 4px;color:#38bdf8;font-size:10px;font-weight:600;
             letter-spacing:.8px;text-transform:uppercase}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 16px;border-radius:6px;
          margin:1px 8px;color:#bae6fd;font-size:13px;font-weight:500;cursor:pointer}
.nav-item:hover{background:var(--nav-hover);color:#fff}
.nav-item.active{background:var(--nav-active);color:#fff}
.sidebar-user{margin-top:auto;padding:12px 16px;border-top:1px solid rgba(255,255,255,.1)}
.user-avatar{width:30px;height:30px;border-radius:50%;background:var(--nav-active);
             display:flex;align-items:center;justify-content:center;
             color:#fff;font-size:11px;font-weight:700;flex-shrink:0}
.main{margin-left:220px;flex:1;display:flex;flex-direction:column}
.topbar{background:var(--card);border-bottom:1px solid var(--border);padding:0 24px;
        height:56px;display:flex;align-items:center;justify-content:space-between;
        position:sticky;top:0;z-index:50}
.topbar-title{font-size:15px;font-weight:600}
.content{padding:24px;flex:1}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;
      padding:20px;margin-bottom:16px}
.card-title{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;
            letter-spacing:.5px;margin-bottom:14px}
.stats{display:grid;gap:12px;margin-bottom:20px}
.stats-4{grid-template-columns:repeat(4,1fr)}
.stats-3{grid-template-columns:repeat(3,1fr)}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.stat-num{font-size:24px;font-weight:700;margin-bottom:2px}
.stat-lbl{font-size:11px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{padding:9px 12px;text-align:left;font-size:10px;font-weight:600;color:var(--muted);
   text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border);
   background:var(--surface)}
td{padding:10px 12px;border-bottom:1px solid #f1f5f9;vertical-align:middle}
tr:hover td{background:#f8fafc}
input,select,textarea{width:100%;padding:8px 11px;border:1px solid var(--border);
  border-radius:6px;font-size:13px;outline:none;font-family:inherit;
  background:var(--card);color:var(--text)}
input:focus,select:focus,textarea:focus{border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(2,132,199,.08)}
.form-group{margin-bottom:14px}
label.lbl{font-size:12px;font-weight:500;color:#374151;display:block;margin-bottom:5px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:6px;
     font-size:13px;font-weight:500;cursor:pointer;border:none;transition:all .15s}
.btn-primary{background:var(--accent);color:#fff}.btn-primary:hover{background:var(--accent-h)}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface)}
.btn-danger{background:#fef2f2;color:var(--danger);border:1px solid #fecaca}
.btn-success{background:#f0fdf4;color:var(--ok);border:1px solid #bbf7d0}
.btn-warn{background:#fffbeb;color:var(--warn);border:1px solid #fde68a}
.btn-sm{padding:5px 10px;font-size:12px}
.badge{display:inline-flex;align-items:center;padding:3px 8px;border-radius:4px;
       font-size:11px;font-weight:600}
.alert{padding:11px 14px;border-radius:6px;font-size:13px;margin-bottom:14px;
       display:flex;align-items:flex-start;gap:8px}
.alert-ok{background:#f0fdf4;border:1px solid #bbf7d0;color:#166534}
.alert-err{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}
.alert-warn{background:#fffbeb;border:1px solid #fde68a;color:#92400e}
.alert-info{background:#eff6ff;border:1px solid #bfdbfe;color:#1e40af}
.pipeline{display:flex;gap:0;margin-bottom:20px}
.pipe-step{flex:1;text-align:center;padding:10px 8px;font-size:11px;font-weight:600;
           border-top:3px solid var(--border);color:var(--muted);position:relative}
.pipe-step.active{border-top-color:var(--accent);color:var(--accent)}
.pipe-step.done{border-top-color:var(--ok);color:var(--ok)}
.pipe-step.danger{border-top-color:var(--danger);color:var(--danger)}
hr.div{border:none;border-top:1px solid var(--border);margin:16px 0}
.tag{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:3px;
     padding:2px 6px;font-size:11px;margin:2px;color:#475569}
.page-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.page-header h1{font-size:18px;font-weight:700}
.comment-box{background:#f8fafc;border:1px solid var(--border);border-radius:6px;padding:12px;
             margin-bottom:8px;font-size:13px}
.comment-meta{font-size:11px;color:var(--muted);margin-bottom:4px}
"""

ICONS = {
    "dashboard": "📊", "new":    "➕", "cases":   "📁",
    "pipeline":  "🔄", "search": "🔍", "admin":   "⚙️",
    "logout":    "→",  "loan":   "💰", "kyc":     "🛡️",
}


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _get_user(request: Request) -> dict | None:
    token = request.cookies.get("loan_session")
    if not token: return None
    return verify_token(token)


def _guard(request: Request) -> dict:
    user = _get_user(request)
    if not user:
        raise HTTPException(302, headers={"Location": "/loan/login"})
    return user


def _can_decide(user: dict) -> bool:
    return user.get("role") in ("manager", "admin")


def _initials(name: str) -> str:
    parts = (name or "?").split()
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else parts[0][-1])).upper()


# ── Shell ─────────────────────────────────────────────────────────────────────

def _shell(title: str, content: str, active: str = "", user: dict = None, alert: str = "") -> str:
    role   = (user or {}).get("role", "officer")
    uname  = (user or {}).get("username", "")
    init   = _initials((user or {}).get("username", "?").title())
    is_mgr = role in ("manager", "admin")

    def nav(key, label, href, show=True):
        if not show: return ""
        cls = "nav-item active" if key == active else "nav-item"
        return f'<a href="{href}" class="{cls}">{ICONS.get(key,"")} {label}</a>'

    sidebar = f"""
    <div class="sidebar">
      <div class="sidebar-logo">
        <div class="title">Loan Management</div>
        <div class="sub">Officer Portal</div>
      </div>
      <div style="overflow-y:auto;flex:1;padding:8px 0">
        <div class="nav-section">Main</div>
        {nav("dashboard","Dashboard","/loan/")}
        {nav("new","New Application","/loan/new")}
        {nav("cases","All Cases","/loan/cases")}
        {nav("pipeline","Pipeline View","/loan/pipeline")}
        <div class="nav-section">Tools</div>
        {nav("search","Search Applicant","/loan/search")}
        {nav("kyc","KYC Dashboard","/loan/kyc-dashboard")}
        {nav("admin","Admin","/loan/admin",is_mgr)}
      </div>
      <div class="sidebar-user">
        <div style="display:flex;align-items:center;gap:10px">
          <div class="user-avatar">{init}</div>
          <div style="flex:1;min-width:0">
            <div style="color:#e0f2fe;font-size:12px;font-weight:600;
                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{uname}</div>
            <div style="color:#7dd3fc;font-size:11px;text-transform:capitalize">{role}</div>
          </div>
          <a href="/loan/logout" style="color:#7dd3fc;font-size:18px" title="Logout">⏏</a>
        </div>
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title} — Loan Portal</title>
  <style>{CSS}</style>
</head>
<body>
  {sidebar}
  <div class="main">
    <div class="topbar">
      <div class="topbar-title">{title}</div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="font-size:12px;color:var(--muted)">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                background:#0284c7;margin-right:4px"></span>{role.title()}
        </span>
        <a href="/loan/logout" class="btn btn-ghost btn-sm">Sign out</a>
      </div>
    </div>
    <div class="content">
      {alert}
      {content}
    </div>
  </div>
</body>
</html>"""


def _alert(msg: str, kind: str = "ok") -> str:
    icons = {"ok": "✓", "err": "✕", "warn": "⚠", "info": "ℹ"}
    cls   = {"ok": "alert-ok", "err": "alert-err", "warn": "alert-warn", "info": "alert-info"}
    return f'<div class="alert {cls.get(kind,"alert-info")}"><span>{icons.get(kind)}</span><span>{msg}</span></div>' if msg else ""


def _status_badge(status: str) -> str:
    color = STATUS_COLOR.get(status, "#94a3b8")
    return f'<span class="badge" style="background:{color}20;color:{color};border:1px solid {color}40">{status.upper()}</span>'


def _risk_badge(level: str) -> str:
    if not level: return "—"
    c = RISK_COLOR.get(level, "#94a3b8")
    a = {"HIGH":"REJECT","MEDIUM":"REVIEW","LOW":"PASS"}.get(level, level)
    return f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}40">{level} — {a}</span>'


def _pipeline_steps(status: str) -> str:
    steps = ["draft","submitted","screening","review","approved"]
    html  = '<div class="pipeline">'
    for s in steps:
        if status == "rejected":
            cls = "pipe-step danger" if s == "review" else ("pipe-step done" if steps.index(s) < steps.index("review") else "pipe-step")
        elif status == s:
            cls = "pipe-step active"
        elif steps.index(s) < steps.index(status) if status in steps else False:
            cls = "pipe-step done"
        else:
            cls = "pipe-step"
        label = s.title()
        html += f'<div class="{cls}">{label}</div>'
    if status == "rejected":
        html += '<div class="pipe-step danger">Rejected</div>'
    elif status == "approved":
        html = html  # already handled
    html += "</div>"
    return html


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    init_loan_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    err = _alert(error, "err") if error else ""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Loan Portal — Sign In</title>
<style>
{CSS}
body{{background:linear-gradient(135deg,#0c4a6e 0%,#0369a1 100%);
     display:flex;align-items:center;justify-content:center;min-height:100vh}}
.login-card{{background:#fff;border-radius:14px;padding:36px 32px;width:360px;
             box-shadow:0 20px 60px rgba(0,0,0,.25)}}
.login-logo{{text-align:center;margin-bottom:24px}}
.icon-wrap{{width:52px;height:52px;background:#e0f2fe;border-radius:12px;
            display:flex;align-items:center;justify-content:center;margin:0 auto 10px;font-size:22px}}
h1{{font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:3px}}
p{{color:#64748b;font-size:13px;margin-bottom:24px}}
.btn-login{{width:100%;padding:10px;background:#0284c7;color:#fff;border:none;
            border-radius:7px;font-size:14px;font-weight:600;cursor:pointer;margin-top:4px}}
.btn-login:hover{{background:#0369a1}}
.hint{{background:#f0f9ff;border-radius:6px;padding:10px 12px;font-size:11px;color:#6b7280;margin-top:14px}}
.hint code{{color:#0284c7;font-weight:600}}
</style></head>
<body>
<div class="login-card">
  <div class="login-logo">
    <div class="icon-wrap">💰</div>
    <h1>Loan Portal</h1>
    <p>Sign in to manage loan applications</p>
  </div>
  {err}
  <form method="post" action="/loan/login">
    <div class="form-group">
      <label class="lbl">Username</label>
      <input type="text" name="username" placeholder="Enter username" required autofocus>
    </div>
    <div class="form-group">
      <label class="lbl">Password</label>
      <input type="password" name="password" placeholder="Enter password" required>
    </div>
    <button type="submit" class="btn-login">Sign In</button>
  </form>
  <div class="hint">
    Officer: <code>officer</code> / <code>officer123</code> &nbsp;·&nbsp;
    Manager: <code>manager</code> / <code>manager123</code>
  </div>
</div>
</body></html>"""
    return HTMLResponse(html)


@app.post("/login")
async def login(username: str = Form(""), password: str = Form("")):
    user = authenticate(username, password)
    if not user:
        return RedirectResponse("/loan/login?error=Invalid+username+or+password", 303)
    token    = make_token(user["username"], user["role"])
    response = RedirectResponse("/loan/", 303)
    response.set_cookie("loan_session", token, httponly=True, samesite="lax", max_age=8*3600)
    return response


@app.get("/logout")
def logout():
    r = RedirectResponse("/loan/login", 303)
    r.delete_cookie("loan_session")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    user  = _guard(request)
    stats = get_stats()

    recent = list_loans(limit=8)
    rows   = ""
    for l in recent:
        amt = f"${float(l.get('loan_amount',0)):,.0f}"
        dt  = str(l.get("created_at",""))[:10]
        rows += (
            "<tr>"
            f"<td><a href='/loan/case/{l['loan_ref']}' style='color:var(--accent);font-weight:500'>{l['loan_ref']}</a></td>"
            f"<td>{l.get('applicant_name','—')}</td>"
            f"<td><span class='tag'>{l.get('loan_type','')}</span></td>"
            f"<td style='font-weight:600'>{amt}</td>"
            f"<td>{_status_badge(l.get('status',''))}</td>"
            f"<td>{_risk_badge(l.get('kyc_risk_level',''))}</td>"
            f"<td style='font-size:12px;color:var(--muted)'>{dt}</td>"
            f"<td><a href='/loan/case/{l['loan_ref']}' class='btn btn-ghost btn-sm'>View</a></td>"
            "</tr>"
        )

    approved_vol = stats.get("approved_volume") or 0

    content = f"""
    <div class="stats stats-4">
      <div class="stat-card"><div class="stat-num">{stats.get("total",0)}</div><div class="stat-lbl">Total Applications (30d)</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#f59e0b">{stats.get("review",0)}</div><div class="stat-lbl">Pending Review</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#22c55e">{stats.get("approved",0)}</div><div class="stat-lbl">Approved</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#ef4444">{stats.get("rejected",0)}</div><div class="stat-lbl">Rejected</div></div>
    </div>
    <div class="stats stats-3">
      <div class="stat-card"><div class="stat-num" style="color:#8b5cf6">{stats.get("screening",0)}</div><div class="stat-lbl">In KYC Screening</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#ef4444">{stats.get("high_risk",0)}</div><div class="stat-lbl">HIGH Risk Flagged</div></div>
      <div class="stat-card"><div class="stat-num" style="color:#22c55e">${float(approved_vol):,.0f}</div><div class="stat-lbl">Approved Volume (30d)</div></div>
    </div>

    <div style="display:flex;gap:10px;margin-bottom:16px">
      <a href="/loan/new" class="btn btn-primary">+ New Application</a>
      <a href="/loan/pipeline" class="btn btn-ghost">Pipeline View</a>
      <a href="/loan/cases?status=review" class="btn btn-warn">⚠ Pending Review ({stats.get("review",0)})</a>
    </div>

    <div class="card">
      <div class="card-title">Recent Applications</div>
      <table>
        <thead><tr><th>Ref</th><th>Applicant</th><th>Type</th><th>Amount</th><th>Status</th><th>KYC</th><th>Date</th><th></th></tr></thead>
        <tbody>{rows or "<tr><td colspan='8' style='text-align:center;padding:20px;color:var(--muted)'>No applications yet</td></tr>"}</tbody>
      </table>
    </div>"""

    return HTMLResponse(_shell("Dashboard", content, "dashboard", user))


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/new", response_class=HTMLResponse)
def new_application_page(request: Request, msg: str = "", error: str = ""):
    user  = _guard(request)
    alert = _alert(msg, "ok") if msg else (_alert(error, "err") if error else "")

    loan_opts = "".join(f"<option>{t}</option>" for t in LOAN_TYPES)

    content = f"""
    {alert}
    <form action="/loan/new" method="post" enctype="multipart/form-data">
      <div class="grid-2">
        <!-- Left: Loan details -->
        <div>
          <div class="card">
            <div class="card-title">Loan Details</div>
            <div class="form-group">
              <label class="lbl">Loan Type *</label>
              <select name="loan_type" required>{loan_opts}</select>
            </div>
            <div class="grid-2">
              <div class="form-group">
                <label class="lbl">Loan Amount (USD) *</label>
                <input type="number" name="loan_amount" min="100" step="100" required placeholder="e.g. 10000">
              </div>
              <div class="form-group">
                <label class="lbl">Term (months) *</label>
                <input type="number" name="loan_term_months" min="1" max="360" value="12" required>
              </div>
            </div>
            <div class="form-group">
              <label class="lbl">Purpose *</label>
              <input type="text" name="loan_purpose" required placeholder="e.g. Working capital, Home renovation">
            </div>
          </div>

          <div class="card">
            <div class="card-title">Applicant Information</div>
            <div class="grid-2">
              <div class="form-group">
                <label class="lbl">Applicant ID *</label>
                <input type="text" name="applicant_id" required placeholder="e.g. APP-001">
              </div>
              <div class="form-group">
                <label class="lbl">Full Name *</label>
                <input type="text" name="applicant_name" required placeholder="Full legal name">
              </div>
              <div class="form-group">
                <label class="lbl">Date of Birth</label>
                <input type="date" name="applicant_dob">
              </div>
              <div class="form-group">
                <label class="lbl">Phone *</label>
                <input type="text" name="applicant_phone" required placeholder="+855 xx xxx xxx">
              </div>
              <div class="form-group">
                <label class="lbl">Email</label>
                <input type="email" name="applicant_email" placeholder="applicant@email.com">
              </div>
              <div class="form-group">
                <label class="lbl">Monthly Income (USD)</label>
                <input type="number" name="applicant_income" min="0" step="10" placeholder="0">
              </div>
            </div>
            <div class="form-group">
              <label class="lbl">Employer / Business</label>
              <input type="text" name="applicant_employer" placeholder="Company or business name">
            </div>
            <div class="form-group">
              <label class="lbl">Address</label>
              <textarea name="applicant_address" rows="2" placeholder="Current address"></textarea>
            </div>
          </div>
        </div>

        <!-- Right: Documents -->
        <div>
          <div class="card">
            <div class="card-title">Supporting Documents</div>
            <div class="alert alert-info"><span>ℹ</span><span>
              Documents will be automatically submitted to the KYC fraud detection system.
              Results appear in the case view within 60 seconds.
            </span></div>
            <div class="form-group">
              <label class="lbl">National ID Card / Passport *</label>
              <input type="file" name="id_card" accept=".jpg,.jpeg,.png" required>
              <div style="font-size:11px;color:var(--muted);margin-top:4px">JPG or PNG — front of ID card</div>
            </div>
            <div class="form-group">
              <label class="lbl">Selfie Photo (for face matching)</label>
              <input type="file" name="selfie" accept=".jpg,.jpeg,.png">
            </div>
            <div class="form-group">
              <label class="lbl">Bank Statement (last 3 months)</label>
              <input type="file" name="bank_statement" accept=".pdf,.jpg,.jpeg,.png">
            </div>
            <div class="form-group">
              <label class="lbl">Payslip / Salary Certificate</label>
              <input type="file" name="payslip" accept=".pdf,.jpg,.jpeg,.png">
            </div>
            <div class="form-group">
              <label class="lbl">Utility Bill (address proof)</label>
              <input type="file" name="utility_bill" accept=".pdf,.jpg,.jpeg,.png">
            </div>
            <div class="form-group">
              <label class="lbl">Other Supporting Document</label>
              <input type="file" name="other_doc" accept=".pdf,.jpg,.jpeg,.png">
            </div>
          </div>

          <div class="card">
            <div class="card-title">Submission</div>
            <p style="font-size:13px;color:var(--muted);margin-bottom:14px">
              Clicking Submit will save the application and immediately send all documents
              to the KYC screening system. You will be redirected to the case view.
            </p>
            <div style="display:flex;gap:10px">
              <button type="submit" name="action" value="submit" class="btn btn-primary">
                Submit & Screen Documents
              </button>
              <button type="submit" name="action" value="draft" class="btn btn-ghost">
                Save as Draft
              </button>
            </div>
          </div>
        </div>
      </div>
    </form>"""

    return HTMLResponse(_shell("New Application", content, "new", user))


@app.post("/new")
async def submit_new_application(
    request:           Request,
    action:            str = Form("submit"),
    loan_type:         str = Form(""),
    loan_amount:       str = Form("0"),
    loan_term_months:  str = Form("12"),
    loan_purpose:      str = Form(""),
    applicant_id:      str = Form(""),
    applicant_name:    str = Form(""),
    applicant_dob:     str = Form(""),
    applicant_phone:   str = Form(""),
    applicant_email:   str = Form(""),
    applicant_income:  str = Form("0"),
    applicant_employer:str = Form(""),
    applicant_address: str = Form(""),
    id_card:           UploadFile = File(None),
    selfie:            UploadFile = File(None),
    bank_statement:    UploadFile = File(None),
    payslip:           UploadFile = File(None),
    utility_bill:      UploadFile = File(None),
    other_doc:         UploadFile = File(None),
):
    user = _guard(request)

    # Create loan record
    loan = create_loan({
        "loan_type":          loan_type,
        "loan_amount":        loan_amount,
        "loan_term_months":   loan_term_months,
        "loan_purpose":       loan_purpose,
        "applicant_id":       applicant_id,
        "applicant_name":     applicant_name,
        "applicant_dob":      applicant_dob,
        "applicant_phone":    applicant_phone,
        "applicant_email":    applicant_email,
        "applicant_income":   applicant_income,
        "applicant_employer": applicant_employer,
        "applicant_address":  applicant_address,
        "created_by":         user["username"],
    })

    if action == "draft":
        return RedirectResponse(f"/loan/case/{loan.loan_ref}?msg=Saved+as+draft", 303)

    # Submit to KYC API
    try:
        files = {}
        docs_uploaded = []

        for field_name, upload in [
            ("id_card", id_card), ("selfie", selfie),
            ("bank_statement", bank_statement), ("payslip", payslip),
            ("utility_bill", utility_bill), ("other_doc", other_doc),
        ]:
            if upload and upload.filename:
                data = await upload.read()
                if data:
                    files[field_name] = (upload.filename, data, upload.content_type or "application/octet-stream")
                    docs_uploaded.append(upload.filename)

        update_loan(loan.loan_ref, {
            "status":        "screening",
            "submitted_at":  datetime.utcnow().isoformat(),
            "kyc_status":    "screening",
            "docs_uploaded": json.dumps(docs_uploaded),
        })

        # Call KYC loan-case API
        headers = {}
        if KYC_API_KEY:
            headers["X-API-Key"] = KYC_API_KEY

        async with httpx.AsyncClient(base_url=KYC_API_URL, timeout=120) as client:
            r = await client.post(
                "/api/v1/loan-case",
                headers = headers,
                data    = {
                    "loan_ref":      loan.loan_ref,
                    "applicant_id":  applicant_id,
                    "applicant_name": applicant_name,
                    "loan_type":     loan_type,
                    "loan_amount":   loan_amount,
                    "source_system": "LoanOfficerPortal",
                    "callback_url":  f"http://loan-portal:8005/webhook/kyc-result",
                },
                files = files if files else {"_dummy": ("x.txt", b"x", "text/plain")},
            )

        if r.status_code == 200:
            result = r.json()
            overall = result.get("overall", {})
            update_loan(loan.loan_ref, {
                "kyc_status":    "complete",
                "kyc_risk_level": overall.get("risk_level", "UNKNOWN"),
                "kyc_risk_score": overall.get("risk_score", 0),
                "kyc_action":    overall.get("action", "REVIEW"),
                "kyc_flags":     json.dumps(overall.get("flags", [])),
                "kyc_result":    json.dumps(result),
                "kyc_screened_at": datetime.utcnow().isoformat(),
                "status":        "review",
            })
            add_comment(loan.loan_ref, "system",
                        f"KYC screening complete — Risk: {overall.get('risk_level')} Score: {overall.get('risk_score')} Action: {overall.get('action')}")
        else:
            update_loan(loan.loan_ref, {"kyc_status": "failed"})
            add_comment(loan.loan_ref, "system", f"KYC screening failed — HTTP {r.status_code}")

    except Exception as e:
        log.error(f"KYC submission failed: {e}")
        update_loan(loan.loan_ref, {"kyc_status": "failed"})
        add_comment(loan.loan_ref, "system", f"KYC submission error: {str(e)[:200]}")

    return RedirectResponse(f"/loan/case/{loan.loan_ref}", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE VIEW
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/case/{loan_ref}", response_class=HTMLResponse)
def case_view(loan_ref: str, request: Request, msg: str = ""):
    user = _guard(request)
    loan = get_loan(loan_ref)
    if not loan:
        raise HTTPException(404, "Loan case not found")

    comments = get_comments(loan_ref)
    alert    = _alert(msg, "ok") if msg else ""

    kyc_result = loan.get("kyc_result") or {}
    if isinstance(kyc_result, str):
        try: kyc_result = json.loads(kyc_result)
        except: kyc_result = {}

    kyc_flags = loan.get("kyc_flags") or []
    if isinstance(kyc_flags, str):
        try: kyc_flags = json.loads(kyc_flags)
        except: kyc_flags = []

    docs = loan.get("docs_uploaded") or []
    if isinstance(docs, str):
        try: docs = json.loads(docs)
        except: docs = []

    # KYC document breakdown
    doc_rows = ""
    documents = kyc_result.get("documents", {})
    for doc_type, doc_data in documents.items():
        if not isinstance(doc_data, dict): continue
        level  = doc_data.get("risk_level", "—")
        score  = doc_data.get("risk_score", "—")
        action = doc_data.get("action", "—")
        fname  = doc_data.get("document", doc_type.replace("_"," ").title())
        dflags = doc_data.get("flags", [])
        c      = RISK_COLOR.get(level, "#94a3b8")
        doc_rows += (
            f"<tr><td style='font-weight:500'>{fname}</td>"
            f"<td style='color:{c};font-weight:600'>{level}</td>"
            f"<td style='text-align:center'>{score}</td>"
            f"<td style='text-align:center'><span class='badge' style='background:{c}20;color:{c}'>{action}</span></td>"
            f"<td style='font-size:11px;color:var(--muted)'>{len(dflags)} flags</td></tr>"
        )

    # Flags
    flag_html = " ".join(f'<span class="tag">{f}</span>' for f in kyc_flags[:20]) or "<span style='color:var(--muted)'>None</span>"

    # Comments
    comment_html = ""
    for c in comments:
        is_system = c["author"] == "system"
        bg        = "#f0f9ff" if is_system else "#f8fafc"
        comment_html += f"""
        <div class="comment-box" style="background:{bg}">
          <div class="comment-meta">{'🤖 System' if is_system else '👤 ' + c['author']} · {c['created_at']}</div>
          <div>{c['comment']}</div>
        </div>"""

    # Decision section
    can_decide = _can_decide(user)
    kyc_action = loan.get("kyc_action", "")
    status     = loan.get("status", "draft")

    decision_html = ""
    if status in ("review", "screening") and can_decide:
        decision_html = f"""
        <div class="card">
          <div class="card-title">Loan Decision</div>
          <div class="alert alert-{'err' if kyc_action=='REJECT' else 'warn' if kyc_action=='REVIEW' else 'ok'}">
            <span>{'✕' if kyc_action=='REJECT' else '⚠' if kyc_action=='REVIEW' else '✓'}</span>
            <span>KYC recommendation: <strong>{kyc_action}</strong> — {loan.get('kyc_risk_level','?')} risk, score {loan.get('kyc_risk_score','?')}</span>
          </div>
          <form method="post" action="/loan/case/{loan_ref}/decide">
            <div class="grid-2">
              <div class="form-group">
                <label class="lbl">Decision *</label>
                <select name="decision" required>
                  <option value="">Select...</option>
                  <option value="approved">Approve Loan</option>
                  <option value="rejected">Reject Loan</option>
                  <option value="review">Request More Information</option>
                </select>
              </div>
              <div class="form-group">
                <label class="lbl">Interest Rate (%)</label>
                <input type="number" name="interest_rate" step="0.1" min="0" max="100" placeholder="e.g. 12.5">
              </div>
            </div>
            <div class="form-group">
              <label class="lbl">Decision Notes *</label>
              <textarea name="notes" rows="3" required placeholder="Explain the decision..."></textarea>
            </div>
            <button type="submit" class="btn btn-primary">Submit Decision</button>
          </form>
        </div>"""
    elif status in ("approved", "rejected"):
        decision_html = f"""
        <div class="card" style="border-left:3px solid {'#22c55e' if status=='approved' else '#ef4444'}">
          <div class="card-title">Decision Recorded</div>
          <div class="grid-2">
            <div><div style="font-size:11px;color:var(--muted)">Decision</div>
                 <div style="font-weight:700;color:{'#22c55e' if status=='approved' else '#ef4444'};font-size:16px">{status.upper()}</div></div>
            <div><div style="font-size:11px;color:var(--muted)">Decided By</div>
                 <div style="font-weight:500">{loan.get('decided_by','—')}</div></div>
            <div><div style="font-size:11px;color:var(--muted)">Date</div>
                 <div>{str(loan.get('decided_at','—'))[:16]}</div></div>
            <div><div style="font-size:11px;color:var(--muted)">Interest Rate</div>
                 <div>{loan.get('interest_rate','—')}%</div></div>
          </div>
          <div style="margin-top:10px"><div style="font-size:11px;color:var(--muted);margin-bottom:4px">Notes</div>
               <div style="font-size:13px">{loan.get('decision_notes','—')}</div></div>
        </div>"""

    content = f"""
    <div class="page-header">
      <div>
        <a href="/loan/cases" style="font-size:12px;color:var(--muted)">← All Cases</a>
        <h1 style="margin-top:4px">{loan_ref} — {loan.get('applicant_name','—')}</h1>
      </div>
      <div style="display:flex;gap:8px">
        {_status_badge(status)}
        {'<a href="/loan/case/'+loan_ref+'/rescreen" class="btn btn-ghost btn-sm">Re-screen KYC</a>' if status not in ('approved','rejected') else ''}
      </div>
    </div>

    {_pipeline_steps(status)}

    <div class="grid-2">
      <div>
        <div class="card">
          <div class="card-title">Loan Information</div>
          <table>
            <tr><td style="color:var(--muted);width:40%">Reference</td><td style="font-weight:600;font-family:monospace">{loan_ref}</td></tr>
            <tr><td style="color:var(--muted)">Loan Type</td><td>{loan.get('loan_type','—')}</td></tr>
            <tr><td style="color:var(--muted)">Amount</td><td style="font-weight:700">${float(loan.get('loan_amount',0)):,.0f}</td></tr>
            <tr><td style="color:var(--muted)">Term</td><td>{loan.get('loan_term_months','—')} months</td></tr>
            <tr><td style="color:var(--muted)">Purpose</td><td>{loan.get('loan_purpose','—')}</td></tr>
            <tr><td style="color:var(--muted)">Submitted</td><td>{str(loan.get('submitted_at','—'))[:16]}</td></tr>
            <tr><td style="color:var(--muted)">Officer</td><td>{loan.get('created_by','—')}</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="card-title">Applicant Information</div>
          <table>
            <tr><td style="color:var(--muted);width:40%">ID</td><td style="font-family:monospace">{loan.get('applicant_id','—')}</td></tr>
            <tr><td style="color:var(--muted)">Name</td><td style="font-weight:500">{loan.get('applicant_name','—')}</td></tr>
            <tr><td style="color:var(--muted)">Date of Birth</td><td>{loan.get('applicant_dob','—')}</td></tr>
            <tr><td style="color:var(--muted)">Phone</td><td>{loan.get('applicant_phone','—')}</td></tr>
            <tr><td style="color:var(--muted)">Email</td><td>{loan.get('applicant_email','—')}</td></tr>
            <tr><td style="color:var(--muted)">Employer</td><td>{loan.get('applicant_employer','—')}</td></tr>
            <tr><td style="color:var(--muted)">Monthly Income</td><td>${float(loan.get('applicant_income',0) or 0):,.0f}</td></tr>
            <tr><td style="color:var(--muted)">Address</td><td style="font-size:12px">{loan.get('applicant_address','—')}</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="card-title">Documents Uploaded ({len(docs)})</div>
          {"".join(f'<div style="padding:6px 0;border-bottom:1px solid #f1f5f9;font-size:13px">📄 {d}</div>' for d in docs) or "<p style='color:var(--muted);font-size:13px'>No documents uploaded</p>"}
        </div>
      </div>

      <div>
        <div class="card" style="border-top:3px solid {RISK_COLOR.get(loan.get('kyc_risk_level',''),'#94a3b8')}">
          <div class="card-title">KYC Screening Result</div>
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
            {_risk_badge(loan.get('kyc_risk_level',''))}
            <span style="font-size:13px;color:var(--muted)">Score: <strong>{loan.get('kyc_risk_score','—')}</strong></span>
            <span style="font-size:12px;color:var(--muted)">Screened: {str(loan.get('kyc_screened_at','—'))[:16]}</span>
          </div>
          {"<table><thead><tr><th>Document</th><th>Risk</th><th>Score</th><th>Action</th><th>Flags</th></tr></thead><tbody>" + doc_rows + "</tbody></table>" if doc_rows else "<p style='color:var(--muted);font-size:13px'>Screening pending or not completed</p>"}
        </div>

        <div class="card">
          <div class="card-title">Fraud Flags ({len(kyc_flags)})</div>
          <div style="line-height:2">{flag_html}</div>
        </div>

        {decision_html}

        <div class="card">
          <div class="card-title">Case Comments</div>
          {comment_html or "<p style='color:var(--muted);font-size:13px'>No comments yet</p>"}
          <hr class="div">
          <form method="post" action="/loan/case/{loan_ref}/comment">
            <div class="form-group">
              <label class="lbl">Add Comment</label>
              <textarea name="comment" rows="2" placeholder="Add a note..." required></textarea>
            </div>
            <button type="submit" class="btn btn-ghost btn-sm">Add Comment</button>
          </form>
        </div>
      </div>
    </div>"""

    return HTMLResponse(_shell(f"Case {loan_ref}", content, "cases", user, alert))


@app.post("/case/{loan_ref}/decide")
async def decide_case(
    loan_ref: str, request: Request,
    decision: str = Form(""), notes: str = Form(""),
    interest_rate: str = Form(""),
):
    user = _guard(request)
    if not _can_decide(user):
        raise HTTPException(403, "Only managers can make decisions")

    status_map = {"approved": "approved", "rejected": "rejected", "review": "review"}
    new_status = status_map.get(decision, "review")

    update_loan(loan_ref, {
        "status":        new_status,
        "decision":      decision,
        "decision_notes": notes,
        "decided_by":    user["username"],
        "decided_at":    datetime.utcnow().isoformat(),
        "interest_rate": float(interest_rate) if interest_rate else None,
    })
    add_comment(loan_ref, user["username"], f"Decision: {decision.upper()} — {notes}")
    return RedirectResponse(f"/loan/case/{loan_ref}?msg=Decision+recorded", 303)


@app.post("/case/{loan_ref}/comment")
async def add_case_comment(loan_ref: str, request: Request, comment: str = Form("")):
    user = _guard(request)
    add_comment(loan_ref, user["username"], comment)
    return RedirectResponse(f"/loan/case/{loan_ref}", 303)


@app.get("/case/{loan_ref}/rescreen")
async def rescreen_case(loan_ref: str, request: Request):
    user = _guard(request)
    loan = get_loan(loan_ref)
    if not loan:
        raise HTTPException(404)
    update_loan(loan_ref, {"kyc_status": "pending", "status": "submitted"})
    add_comment(loan_ref, user["username"], "Re-screening requested")
    return RedirectResponse(f"/loan/case/{loan_ref}?msg=Re-screening+requested", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  CASES LIST
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/cases", response_class=HTMLResponse)
def cases_list(request: Request, status: str = "", msg: str = ""):
    user  = _guard(request)
    loans = list_loans(status=status, limit=100)

    rows = ""
    for l in loans:
        amt = f"${float(l.get('loan_amount',0)):,.0f}"
        dt  = str(l.get("created_at",""))[:10]
        rows += (
            f"<tr>"
            f"<td><a href='/loan/case/{l['loan_ref']}' style='color:var(--accent);font-weight:500'>{l['loan_ref']}</a></td>"
            f"<td>{l.get('applicant_name','—')}</td>"
            f"<td>{l.get('applicant_id','—')}</td>"
            f"<td><span class='tag'>{l.get('loan_type','')}</span></td>"
            f"<td style='font-weight:600'>{amt}</td>"
            f"<td>{_status_badge(l.get('status',''))}</td>"
            f"<td>{_risk_badge(l.get('kyc_risk_level',''))}</td>"
            f"<td style='font-size:12px;color:var(--muted)'>{dt}</td>"
            f"<td><a href='/loan/case/{l['loan_ref']}' class='btn btn-ghost btn-sm'>View</a></td>"
            f"</tr>"
        )

    statuses = ["", "draft", "submitted", "screening", "review", "approved", "rejected"]
    filter_tabs = ""
    for s in statuses:
        label = s.title() if s else "All"
        active = "btn-primary" if status == s else "btn-ghost"
        filter_tabs += f'<a href="/loan/cases{"?status="+s if s else ""}" class="btn {active} btn-sm">{label}</a>'

    content = f"""
    <div class="page-header">
      <h1>All Loan Cases</h1>
      <a href="/loan/new" class="btn btn-primary">+ New Application</a>
    </div>
    <div style="display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap">{filter_tabs}</div>
    <div class="card">
      <div class="card-title">{len(loans)} case(s){' — ' + status.title() if status else ''}</div>
      <table>
        <thead><tr><th>Ref</th><th>Applicant</th><th>ID</th><th>Type</th><th>Amount</th><th>Status</th><th>KYC</th><th>Date</th><th></th></tr></thead>
        <tbody>{rows or "<tr><td colspan='9' style='text-align:center;padding:20px;color:var(--muted)'>No cases found</td></tr>"}</tbody>
      </table>
    </div>"""

    return HTMLResponse(_shell("Cases", content, "cases", user, _alert(msg,"ok") if msg else ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE VIEW
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/pipeline", response_class=HTMLResponse)
def pipeline_view(request: Request):
    user = _guard(request)

    columns = {
        "draft":     list_loans(status="draft",     limit=20),
        "screening": list_loans(status="screening", limit=20),
        "review":    list_loans(status="review",    limit=20),
        "approved":  list_loans(status="approved",  limit=10),
        "rejected":  list_loans(status="rejected",  limit=10),
    }

    def card(l):
        risk   = l.get("kyc_risk_level","")
        rc     = RISK_COLOR.get(risk,"#94a3b8")
        amt    = f"${float(l.get('loan_amount',0)):,.0f}"
        return f"""
        <a href="/loan/case/{l['loan_ref']}" style="display:block;background:#fff;border:1px solid var(--border);
           border-radius:8px;padding:12px;margin-bottom:8px;border-left:3px solid {rc}">
          <div style="font-weight:600;font-size:13px;color:var(--accent)">{l['loan_ref']}</div>
          <div style="font-size:12px;color:var(--text);margin:3px 0">{l.get('applicant_name','—')}</div>
          <div style="display:flex;justify-content:space-between;margin-top:6px">
            <span style="font-size:11px;font-weight:600">{amt}</span>
            {"<span style='font-size:10px;font-weight:600;color:" + rc + "'>" + risk + "</span>" if risk else ""}
          </div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px">{l.get('loan_type','')}</div>
        </a>"""

    cols_html = ""
    labels    = {"draft":"Draft","screening":"KYC Screening","review":"Pending Review","approved":"Approved","rejected":"Rejected"}
    border    = {"draft":"#94a3b8","screening":"#8b5cf6","review":"#f59e0b","approved":"#22c55e","rejected":"#ef4444"}

    for key, items in columns.items():
        cards = "".join(card(l) for l in items) or f"<div style='font-size:12px;color:var(--muted);padding:8px'>No cases</div>"
        cols_html += f"""
        <div style="min-width:200px;flex:1">
          <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
               color:{border[key]};border-top:3px solid {border[key]};padding-top:8px;margin-bottom:10px">
            {labels[key]} ({len(items)})
          </div>
          {cards}
        </div>"""

    content = f"""
    <div class="page-header">
      <h1>Pipeline View</h1>
      <a href="/loan/new" class="btn btn-primary">+ New Application</a>
    </div>
    <div style="display:flex;gap:16px;overflow-x:auto;padding-bottom:16px">{cols_html}</div>"""

    return HTMLResponse(_shell("Pipeline", content, "pipeline", user))


# ═══════════════════════════════════════════════════════════════════════════════
#  SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = ""):
    user  = _guard(request)
    rows  = ""
    loans = []

    if q:
        from sqlalchemy import text
        db = SessionLocal()
        try:
            results = db.execute(text("""
                SELECT loan_ref, loan_type, loan_amount, applicant_name, applicant_id,
                       applicant_phone, status, kyc_risk_level, created_at
                FROM loan_applications
                WHERE applicant_name ILIKE :q OR applicant_id ILIKE :q
                   OR loan_ref ILIKE :q OR applicant_phone ILIKE :q
                ORDER BY created_at DESC LIMIT 50
            """), {"q": f"%{q}%"}).fetchall()
            loans = [dict(r._mapping) for r in results]
        finally:
            db.close()

        for l in loans:
            amt = f"${float(l.get('loan_amount',0)):,.0f}"
            dt  = str(l.get("created_at",""))[:10]
            rows += (
                f"<tr>"
                f"<td><a href='/loan/case/{l['loan_ref']}' style='color:var(--accent);font-weight:500'>{l['loan_ref']}</a></td>"
                f"<td>{l.get('applicant_name','—')}</td>"
                f"<td>{l.get('applicant_id','—')}</td>"
                f"<td>{l.get('applicant_phone','—')}</td>"
                f"<td>{amt}</td>"
                f"<td>{_status_badge(l.get('status',''))}</td>"
                f"<td>{_risk_badge(l.get('kyc_risk_level',''))}</td>"
                f"<td style='font-size:12px;color:var(--muted)'>{dt}</td>"
                f"</tr>"
            )

    content = f"""
    <form method="get" style="margin-bottom:20px">
      <div style="display:flex;gap:10px">
        <input type="text" name="q" value="{q}" placeholder="Search by name, applicant ID, reference, or phone..." style="flex:1;padding:10px 14px">
        <button type="submit" class="btn btn-primary">Search</button>
      </div>
    </form>
    {"<div class='card'><div class='card-title'>" + str(len(loans)) + " result(s) for: " + q + "</div><table><thead><tr><th>Ref</th><th>Name</th><th>ID</th><th>Phone</th><th>Amount</th><th>Status</th><th>KYC</th><th>Date</th></tr></thead><tbody>" + rows + "</tbody></table></div>" if q else "<div class='card'><p style='color:var(--muted);font-size:13px'>Enter a name, applicant ID, loan reference, or phone number to search.</p></div>"}"""

    return HTMLResponse(_shell("Search Applicant", content, "search", user))


# ═══════════════════════════════════════════════════════════════════════════════
#  KYC DASHBOARD LINK
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/kyc-dashboard", response_class=HTMLResponse)
def kyc_dashboard(request: Request):
    user    = _guard(request)
    kyc_url = os.getenv("KYC_PORTAL_URL", "http://172.16.26.48:3000/dashboard/")
    content = f"""
    <div class="card" style="text-align:center;padding:40px">
      <div style="font-size:48px;margin-bottom:16px">🛡️</div>
      <div style="font-size:18px;font-weight:700;margin-bottom:8px">KYC Fraud Detection System</div>
      <div style="color:var(--muted);font-size:13px;margin-bottom:24px">
        Access the full KYC analytics dashboard and fraud detection portal
      </div>
      <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
        <a href="{kyc_url}" target="_blank" class="btn btn-primary">Open KYC Dashboard</a>
        <a href="http://172.16.26.48:3000/portal/" target="_blank" class="btn btn-ghost">KYC Staff Portal</a>
        <a href="http://172.16.26.48:3000/docs" target="_blank" class="btn btn-ghost">API Documentation</a>
      </div>
    </div>"""
    return HTMLResponse(_shell("KYC Dashboard", content, "kyc", user))


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, msg: str = "", error: str = ""):
    user = _guard(request)
    if not _can_decide(user):
        raise HTTPException(403, "Manager access required")

    officers = get_all_officers()
    rows     = ""
    for o in officers:
        active  = o.get("active", True)
        role    = o.get("role", "officer")
        ll      = str(o.get("last_login",""))[:16] or "Never"
        rc      = {"admin":"#7c3aed","manager":"#0284c7","officer":"#475569"}.get(role,"#475569")
        rows += (
            f"<tr>"
            f"<td style='font-weight:500'>{o.get('username','')}</td>"
            f"<td>{o.get('name','')}</td>"
            f"<td><span class='badge' style='background:{rc}20;color:{rc}'>{role}</span></td>"
            f"<td>{'<span style=\"color:#22c55e;font-weight:600\">Active</span>' if active else '<span style=\"color:#ef4444\">Inactive</span>'}</td>"
            f"<td style='font-size:12px;color:var(--muted)'>{ll}</td>"
            f"</tr>"
        )

    alert = _alert(msg,"ok") if msg else (_alert(error,"err") if error else "")

    content = f"""
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Loan Officers</div>
        <table style="margin-bottom:16px">
          <thead><tr><th>Username</th><th>Name</th><th>Role</th><th>Status</th><th>Last Login</th></tr></thead>
          <tbody>{rows or "<tr><td colspan='5' style='text-align:center;padding:20px;color:var(--muted)'>No officers</td></tr>"}</tbody>
        </table>
        <hr class="div">
        <div class="card-title">Add New Officer</div>
        <form method="post" action="/loan/admin/create-officer">
          <div class="grid-2">
            <div class="form-group"><label class="lbl">Username *</label><input name="username" required></div>
            <div class="form-group"><label class="lbl">Password *</label><input type="password" name="password" required></div>
            <div class="form-group"><label class="lbl">Full Name *</label><input name="name" required></div>
            <div class="form-group">
              <label class="lbl">Role</label>
              <select name="role">
                <option value="officer">Officer — submit and view</option>
                <option value="manager">Manager — submit, view, and decide</option>
                <option value="admin">Admin — full access</option>
              </select>
            </div>
          </div>
          <div class="form-group"><label class="lbl">Email</label><input type="email" name="email"></div>
          <button type="submit" class="btn btn-primary">Create Officer</button>
        </form>
      </div>

      <div class="card">
        <div class="card-title">Change Password</div>
        <form method="post" action="/loan/admin/change-password">
          <div class="form-group"><label class="lbl">Username</label><input name="username" required></div>
          <div class="form-group"><label class="lbl">New Password</label><input type="password" name="new_password" required></div>
          <button type="submit" class="btn btn-ghost">Update Password</button>
        </form>
        <hr class="div">
        <div class="card-title">System Info</div>
        <div class="alert alert-info"><span>ℹ</span><span>
          KYC API: <code>{KYC_API_URL}</code><br>
          API Key: <code>{'Configured' if KYC_API_KEY else 'Not set (API_AUTH_ENABLED=false)'}</code>
        </span></div>
      </div>
    </div>"""

    return HTMLResponse(_shell("Admin", content, "admin", user, alert))


@app.post("/admin/create-officer")
async def create_officer(request: Request,
    username: str=Form(""), password: str=Form(""),
    role: str=Form("officer"), name: str=Form(""), email: str=Form("")):
    user = _guard(request)
    if not _can_decide(user):
        raise HTTPException(403)
    db = SessionLocal()
    try:
        db.add(LoanOfficer(username=username, password=_hash(password),
                            role=role, name=name, email=email))
        db.commit()
        return RedirectResponse(f"/loan/admin?msg=Officer+{username}+created", 303)
    except Exception as e:
        db.rollback()
        return RedirectResponse("/loan/admin?error=Username+already+exists", 303)
    finally:
        db.close()


@app.post("/admin/change-password")
async def change_password(request: Request, username: str=Form(""), new_password: str=Form("")):
    user = _guard(request)
    if not _can_decide(user):
        raise HTTPException(403)
    from sqlalchemy import text
    db = SessionLocal()
    try:
        db.execute(text("UPDATE loan_officers SET password=:p WHERE username=:u"),
                   {"p": _hash(new_password), "u": username})
        db.commit()
        return RedirectResponse("/loan/admin?msg=Password+updated", 303)
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBHOOK — receive KYC results
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/kyc-result")
async def kyc_webhook(request: Request):
    try:
        data     = await request.json()
        loan_ref = data.get("loan_ref")
        if not loan_ref:
            return {"received": True}

        overall = data.get("overall", {})
        update_loan(loan_ref, {
            "kyc_status":     "complete",
            "kyc_risk_level": overall.get("risk_level", "UNKNOWN"),
            "kyc_risk_score": overall.get("risk_score", 0),
            "kyc_action":     overall.get("action", "REVIEW"),
            "kyc_flags":      json.dumps(overall.get("flags", [])),
            "kyc_result":     json.dumps(data),
            "kyc_screened_at": datetime.utcnow().isoformat(),
            "status":         "review",
        })
        add_comment(loan_ref, "system",
                    f"KYC webhook received — Risk: {overall.get('risk_level')} Score: {overall.get('risk_score')}")
        log.info(f"KYC webhook received for {loan_ref}")
    except Exception as e:
        log.error(f"Webhook error: {e}")
    return {"received": True}


@app.get("/health")
def health():
    return {"status": "running", "service": "loan-portal", "version": "1.0.0"}
