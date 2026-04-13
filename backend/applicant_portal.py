"""
applicant_portal.py — Customer-Facing Applicant Portal
Bilingual (Khmer + English), phone OTP or email/password login.
Applicants submit loan applications and track status.
Runs on port 8006, accessible at /apply/ via Nginx.
"""
import os
import re
import json
import random
import string
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response as FResponse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("applicant_portal")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

KYC_API_URL  = os.getenv("KYC_API_URL", "http://api:8001")
SERVER_IP    = os.getenv("SERVER_IP", "172.16.26.48")
SMS_ENABLED  = os.getenv("SMS_ENABLED", "false").lower() == "true"

app = FastAPI(title="Loan Application Portal", version="1.0.0")

# ── Translations ──────────────────────────────────────────────────────────────
T = {
    "en": {
        "title":          "Loan Application Portal",
        "subtitle":       "Apply for a loan quickly and securely",
        "login":          "Sign In",
        "register":       "Apply Now",
        "phone":          "Phone Number",
        "email":          "Email Address",
        "password":       "Password",
        "otp":            "Verification Code",
        "send_otp":       "Send Code",
        "verify":         "Verify",
        "name":           "Full Name",
        "dob":            "Date of Birth",
        "address":        "Current Address",
        "employer":       "Employer / Business",
        "income":         "Monthly Income (USD)",
        "loan_type":      "Loan Type",
        "loan_amount":    "Loan Amount (USD)",
        "loan_term":      "Loan Term (months)",
        "loan_purpose":   "Purpose",
        "id_card":        "National ID Card / Passport",
        "selfie":         "Your Photo (Selfie)",
        "bank_statement": "Bank Statement (last 3 months)",
        "payslip":        "Payslip or Salary Certificate",
        "utility_bill":   "Utility Bill",
        "submit":         "Submit Application",
        "save_draft":     "Save as Draft",
        "status":         "Application Status",
        "ref":            "Reference Number",
        "submitted":      "Submitted",
        "screening":      "Under Review",
        "approved":       "Approved",
        "rejected":       "Not Approved",
        "draft":          "Draft",
        "review":         "Under Review",
        "track":          "Track Application",
        "logout":         "Sign Out",
        "my_apps":        "My Applications",
        "new_app":        "New Application",
        "dashboard":      "Dashboard",
        "welcome":        "Welcome",
        "no_apps":        "No applications yet. Start your first application!",
        "amount_label":   "Loan Amount",
        "type_label":     "Loan Type",
        "date_label":     "Date Applied",
        "decision_notes": "Decision Notes",
        "upload_docs":    "Upload Documents",
        "docs_help":      "Upload clear photos or scans. Max 10MB each.",
        "personal_info":  "Personal Information",
        "loan_details":   "Loan Details",
        "required":       "* Required",
    },
    "km": {
        "title":          "វិបផតថលដាក់ពាក្យខ្ចីប្រាក់",
        "subtitle":       "ដាក់ពាក្យខ្ចីប្រាក់ដោយរហ័ស និងសុវត្ថិភាព",
        "login":          "ចូលប្រើប្រាស់",
        "register":       "ដាក់ពាក្យឥឡូវ",
        "phone":          "លេខទូរស័ព្ទ",
        "email":          "អ៊ីមែល",
        "password":       "ពាក្យសម្ងាត់",
        "otp":            "លេខផ្ទៀងផ្ទាត់",
        "send_otp":       "ផ្ញើលេខ",
        "verify":         "ផ្ទៀងផ្ទាត់",
        "name":           "ឈ្មោះពេញ",
        "dob":            "ថ្ងៃខែឆ្នាំកំណើត",
        "address":        "អាសយដ្ឋានបច្ចុប្បន្ន",
        "employer":       "និយោជក / អាជីវកម្ម",
        "income":         "ប្រាក់ចំណូលប្រចាំខែ (ដុល្លារ)",
        "loan_type":      "ប្រភេទប្រាក់កម្ចី",
        "loan_amount":    "ចំនួនប្រាក់កម្ចី (ដុល្លារ)",
        "loan_term":      "រយៈពេល (ខែ)",
        "loan_purpose":   "គោលបំណង",
        "id_card":        "អត្តសញ្ញាណប័ណ្ណ / លិខិតឆ្លងដែន",
        "selfie":         "រូបថតខ្លួនឯង",
        "bank_statement": "សៀវភៅធនាគារ (៣ខែចុងក្រោយ)",
        "payslip":        "វិក្កយបត្រប្រាក់ខែ",
        "utility_bill":   "វិក្កយបត្រអគ្គិសនី / ទឹក",
        "submit":         "ដាក់ពាក្យ",
        "save_draft":     "រក្សាទុកជាព្រាង",
        "status":         "ស្ថានភាពពាក្យសុំ",
        "ref":            "លេខយោង",
        "submitted":      "បានដាក់ពាក្យ",
        "screening":      "កំពុងពិនិត្យ",
        "approved":       "អនុម័ត",
        "rejected":       "មិនអនុម័ត",
        "draft":          "ព្រាង",
        "review":         "កំពុងពិនិត្យ",
        "track":          "តាមដានពាក្យសុំ",
        "logout":         "ចាកចេញ",
        "my_apps":        "ពាក្យសុំរបស់ខ្ញុំ",
        "new_app":        "ពាក្យសុំថ្មី",
        "dashboard":      "ទំព័រដើម",
        "welcome":        "សូមស្វាគមន៍",
        "no_apps":        "មិនទាន់មានពាក្យសុំ។ ចាប់ផ្តើមពាក្យសុំដំបូងរបស់អ្នក!",
        "amount_label":   "ចំនួនប្រាក់",
        "type_label":     "ប្រភេទ",
        "date_label":     "កាលបរិច្ឆេទ",
        "decision_notes": "កំណត់ចំណាំការសម្រេច",
        "upload_docs":    "បង្ហោះឯកសារ",
        "docs_help":      "បង្ហោះរូបថតឬស្កេនច្បាស់លាស់។ អតិបរមា ១០MB ម្តងៗ។",
        "personal_info":  "ព័ត៌មានផ្ទាល់ខ្លួន",
        "loan_details":   "ព័ត៌មានប្រាក់កម្ចី",
        "required":       "* ចាំបាច់",
    }
}

LOAN_TYPES = {
    "en": ["Personal Loan", "Business / SME Loan", "Mortgage / Home Loan",
           "Vehicle Loan", "Education Loan", "Agricultural Loan"],
    "km": ["ប្រាក់កម្ចីផ្ទាល់ខ្លួន", "ប្រាក់កម្ចីអាជីវកម្ម", "ប្រាក់កម្ចីទិញផ្ទះ",
           "ប្រាក់កម្ចីទិញយានយន្ត", "ប្រាក់កម្ចីសិក្សា", "ប្រាក់កម្ចីកសិកម្ម"],
}

STATUS_COLORS = {
    "draft":     "#94a3b8",
    "submitted": "#3b82f6",
    "screening": "#8b5cf6",
    "review":    "#f59e0b",
    "approved":  "#22c55e",
    "rejected":  "#ef4444",
}


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
:root{--primary:#dc2626;--primary-h:#b91c1c;--surface:#fff7f7;--card:#fff;
      --border:#fecaca;--text:#1a0a0a;--muted:#78716c;--ok:#16a34a;
      --warn:#d97706;--info:#1d4ed8}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans Khmer','Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);min-height:100vh}
.topbar{background:var(--primary);color:#fff;padding:0 20px;height:56px;
        display:flex;align-items:center;justify-content:space-between;
        position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(220,38,38,.3)}
.topbar-brand{font-size:15px;font-weight:700;display:flex;align-items:center;gap:8px}
.topbar-right{display:flex;align-items:center;gap:10px}
.lang-btn{background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.3);
           padding:4px 10px;border-radius:4px;font-size:12px;cursor:pointer;font-weight:600}
.lang-btn:hover{background:rgba(255,255,255,.25)}
.nav-link{color:rgba(255,255,255,.85);font-size:13px;font-weight:500;padding:6px 10px;
           border-radius:4px;cursor:pointer}
.nav-link:hover{background:rgba(255,255,255,.15);color:#fff}
.container{max-width:720px;margin:0 auto;padding:24px 16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:24px;margin-bottom:16px;box-shadow:0 1px 4px rgba(220,38,38,.06)}
.card-title{font-size:13px;font-weight:700;color:var(--primary);text-transform:uppercase;
            letter-spacing:.5px;margin-bottom:16px;padding-bottom:10px;
            border-bottom:1px solid var(--border)}
.form-group{margin-bottom:14px}
label.lbl{font-size:13px;font-weight:600;color:#374151;display:block;margin-bottom:5px}
input,select,textarea{width:100%;padding:10px 13px;border:1.5px solid #e5e7eb;
  border-radius:8px;font-size:14px;outline:none;font-family:inherit;
  background:#fff;color:var(--text);transition:border .15s}
input:focus,select:focus,textarea:focus{border-color:var(--primary);
  box-shadow:0 0 0 3px rgba(220,38,38,.08)}
input[type=file]{padding:8px;background:#fef2f2;border-color:var(--border)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;
     padding:11px 22px;border-radius:8px;font-size:14px;font-weight:600;
     cursor:pointer;border:none;transition:all .15s;width:100%;margin-top:6px}
.btn-primary{background:var(--primary);color:#fff}
.btn-primary:hover{background:var(--primary-h)}
.btn-ghost{background:#f9fafb;color:#374151;border:1.5px solid #e5e7eb}
.btn-ghost:hover{background:#f3f4f6}
.btn-sm{width:auto;padding:7px 16px;font-size:13px;margin-top:0}
.alert{padding:12px 16px;border-radius:8px;font-size:13px;margin-bottom:14px;
       display:flex;align-items:flex-start;gap:8px;line-height:1.5}
.alert-ok{background:#f0fdf4;border:1px solid #bbf7d0;color:#166534}
.alert-err{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}
.alert-info{background:#eff6ff;border:1px solid #bfdbfe;color:#1e40af}
.alert-warn{background:#fffbeb;border:1px solid #fde68a;color:#92400e}
.status-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;
             border-radius:20px;font-size:12px;font-weight:700}
.app-card{background:var(--card);border:1px solid var(--border);border-radius:12px;
          padding:16px 18px;margin-bottom:12px;cursor:pointer;transition:all .15s;
          display:flex;align-items:center;justify-content:space-between}
.app-card:hover{border-color:var(--primary);box-shadow:0 2px 8px rgba(220,38,38,.1)}
.app-ref{font-weight:700;font-size:14px;color:var(--primary);font-family:monospace}
.app-meta{font-size:12px;color:var(--muted);margin-top:3px}
.progress-bar{height:6px;background:#fee2e2;border-radius:3px;overflow:hidden;margin:14px 0}
.progress-fill{height:100%;background:var(--primary);border-radius:3px;transition:width .5s}
.step-row{display:flex;justify-content:space-between;margin-bottom:20px}
.step{text-align:center;flex:1;position:relative}
.step-dot{width:28px;height:28px;border-radius:50%;background:#fee2e2;border:2px solid var(--border);
          display:flex;align-items:center;justify-content:center;margin:0 auto 6px;
          font-size:11px;font-weight:700;color:var(--muted)}
.step.done .step-dot{background:var(--ok);border-color:var(--ok);color:#fff}
.step.active .step-dot{background:var(--primary);border-color:var(--primary);color:#fff}
.step-label{font-size:10px;color:var(--muted);font-weight:500}
.step.active .step-label{color:var(--primary);font-weight:700}
.step.done .step-label{color:var(--ok)}
.divider{border:none;border-top:1px solid var(--border);margin:16px 0}
.hero{text-align:center;padding:40px 20px 20px}
.hero-icon{font-size:48px;margin-bottom:12px}
.hero h1{font-size:22px;font-weight:700;color:var(--primary);margin-bottom:6px}
.hero p{font-size:14px;color:var(--muted);line-height:1.6;max-width:400px;margin:0 auto}
.tab-row{display:flex;gap:0;margin-bottom:20px;border:1.5px solid var(--border);
         border-radius:8px;overflow:hidden}
.tab{flex:1;padding:10px;text-align:center;font-size:13px;font-weight:600;
     cursor:pointer;background:#fff;color:var(--muted);border:none;transition:all .15s}
.tab.active{background:var(--primary);color:#fff}
"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def _q(sql, params={}):
    db = SessionLocal()
    try:
        return [dict(r._mapping) for r in db.execute(text(sql), params)]
    except Exception as e:
        log.error(f"Query: {e}"); return []
    finally:
        db.close()


def _exec(sql, params={}):
    db = SessionLocal()
    try:
        db.execute(text(sql), params)
        db.commit()
    except Exception as e:
        db.rollback(); log.error(f"Exec: {e}")
    finally:
        db.close()


def _init_tables():
    db = SessionLocal()
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS applicants (
                id           SERIAL PRIMARY KEY,
                phone        VARCHAR(20) UNIQUE,
                email        VARCHAR(100) UNIQUE,
                password     VARCHAR(64),
                name         VARCHAR(200),
                dob          VARCHAR(20),
                address      TEXT,
                employer     VARCHAR(200),
                income       FLOAT,
                otp_code     VARCHAR(6),
                otp_expires  TIMESTAMP,
                verified     BOOLEAN DEFAULT FALSE,
                created_at   TIMESTAMP DEFAULT NOW()
            )
        """))
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS applicant_loans (
                id              SERIAL PRIMARY KEY,
                loan_ref        VARCHAR(50) UNIQUE,
                applicant_id    INTEGER REFERENCES applicants(id),
                loan_type       VARCHAR(100),
                loan_amount     FLOAT,
                loan_term       INTEGER,
                loan_purpose    TEXT,
                docs_uploaded   JSON,
                kyc_status      VARCHAR(20) DEFAULT 'pending',
                kyc_risk_level  VARCHAR(10),
                kyc_risk_score  INTEGER,
                kyc_action      VARCHAR(10),
                kyc_flags       JSON,
                kyc_result      JSON,
                status          VARCHAR(20) DEFAULT 'draft',
                decision_notes  TEXT,
                created_at      TIMESTAMP DEFAULT NOW(),
                updated_at      TIMESTAMP DEFAULT NOW(),
                submitted_at    TIMESTAMP
            )
        """))
        db.commit()
    except Exception as e:
        db.rollback(); log.error(f"Init tables: {e}")
    finally:
        db.close()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()


def _gen_otp(): return "".join(random.choices(string.digits, k=6))


def _gen_ref():
    return "AP" + datetime.utcnow().strftime("%y%m%d") + "".join(random.choices(string.digits, k=4))


def _send_otp(phone: str, otp: str):
    """Send OTP via SMS. Replace with your SMS provider."""
    log.info(f"OTP for {phone}: {otp}")  # In production: call Twilio/SMS gateway
    if SMS_ENABLED:
        pass  # add SMS provider call here
    return True


def _get_session(request: Request) -> dict | None:
    import base64, hmac, json, hashlib
    token = request.cookies.get("applicant_session")
    if not token: return None
    try:
        secret = os.getenv("JWT_SECRET", "changeme")
        parts  = token.split(".")
        if len(parts) != 2: return None
        encoded, sig = parts
        expected = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected): return None
        return json.loads(base64.b64decode(encoded).decode())
    except: return None


def _make_session(applicant_id: int, name: str) -> str:
    import base64, hmac, json, hashlib
    secret  = os.getenv("JWT_SECRET", "changeme")
    payload = json.dumps({"id": applicant_id, "name": name,
                          "exp": (datetime.utcnow() + timedelta(hours=24)).isoformat()})
    encoded = base64.b64encode(payload.encode()).decode()
    sig     = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return encoded + "." + sig


def _guard(request: Request) -> dict:
    user = _get_session(request)
    if not user:
        raise HTTPException(302, headers={"Location": "/apply/login"})
    return user


# ── Language helper ────────────────────────────────────────────────────────────

def _lang(request: Request) -> str:
    return request.cookies.get("lang", "en")


def _t(request: Request, key: str) -> str:
    lang = _lang(request)
    return T.get(lang, T["en"]).get(key, T["en"].get(key, key))


# ── Shell ──────────────────────────────────────────────────────────────────────

def _shell(request: Request, content: str, title_key: str = "title",
           user: dict = None, alert: str = "") -> str:
    lang     = _lang(request)
    t        = T.get(lang, T["en"])
    other    = "km" if lang == "en" else "en"
    lang_lbl = "ភាសាខ្មែរ" if lang == "en" else "English"

    nav = ""
    if user:
        nav = f"""
        <a href="/apply/" class="nav-link">{t['dashboard']}</a>
        <a href="/apply/new" class="nav-link">{t['new_app']}</a>
        <a href="/apply/logout" class="nav-link">{t['logout']}</a>"""

    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{t[title_key]} — {t['title']}</title>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Khmer:wght@400;600;700&display=swap" rel="stylesheet">
  <style>{CSS}</style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-brand">
      <span>🏦</span>
      <span>{t['title']}</span>
    </div>
    <div class="topbar-right">
      {nav}
      <form method="post" action="/apply/set-lang" style="display:inline">
        <input type="hidden" name="lang" value="{other}">
        <button type="submit" class="lang-btn">{lang_lbl}</button>
      </form>
    </div>
  </div>
  <div class="container">
    {alert}
    {content}
  </div>
</body>
</html>"""


def _alert(msg, kind="ok"):
    icons = {"ok":"✓","err":"✕","warn":"⚠","info":"ℹ"}
    cls   = {"ok":"alert-ok","err":"alert-err","warn":"alert-warn","info":"alert-info"}
    return f'<div class="alert {cls.get(kind,"alert-info")}"><span>{icons.get(kind)}</span><span>{msg}</span></div>' if msg else ""


def _status_pill(status, lang="en"):
    t   = T.get(lang, T["en"])
    lbl = t.get(status, status)
    c   = STATUS_COLORS.get(status, "#94a3b8")
    return f'<span class="status-pill" style="background:{c}20;color:{c};border:1px solid {c}40">{lbl}</span>'


def _progress(status):
    steps   = ["submitted","screening","review","approved"]
    pct_map = {"draft":5,"submitted":25,"screening":50,"review":75,"approved":100,"rejected":100}
    pct     = pct_map.get(status, 0)
    color   = "#ef4444" if status == "rejected" else "#dc2626"
    return f'<div class="progress-bar"><div class="progress-fill" style="width:{pct}%;background:{color}"></div></div>'


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    _init_tables()


# ═══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/set-lang")
async def set_lang(request: Request, lang: str = Form("en")):
    ref = request.headers.get("referer", "/apply/")
    r   = RedirectResponse(ref, 303)
    r.set_cookie("lang", lang if lang in ("en","km") else "en",
                 max_age=30*24*3600, samesite="lax")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN / REGISTER
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = "", tab: str = "phone"):
    t    = T.get(_lang(request), T["en"])
    err  = _alert(error, "err") if error else ""

    content = f"""
    <div class="hero">
      <div class="hero-icon">🏦</div>
      <h1>{t['title']}</h1>
      <p>{t['subtitle']}</p>
    </div>

    <div class="card">
      <div class="tab-row">
        <button class="tab {'active' if tab=='phone' else ''}"
                onclick="showTab('phone')" id="tab-phone">{t['phone']}</button>
        <button class="tab {'active' if tab=='email' else ''}"
                onclick="showTab('email')" id="tab-email">{t['email']}</button>
      </div>

      <!-- Phone OTP tab -->
      <div id="pane-phone" {'style="display:none"' if tab=='email' else ''}>
        <div class="form-group">
          <label class="lbl">{t['phone']}</label>
          <div style="display:flex;gap:8px">
            <input type="tel" id="phone-input" placeholder="+855 xx xxx xxx" style="flex:1">
            <button class="btn btn-ghost btn-sm" onclick="sendOTP()">{t['send_otp']}</button>
          </div>
        </div>
        <div class="form-group" id="otp-box" style="display:none">
          <label class="lbl">{t['otp']}</label>
          <input type="text" id="otp-input" placeholder="000000" maxlength="6"
                 style="letter-spacing:8px;font-size:20px;text-align:center">
        </div>
        <div id="otp-msg"></div>
        <button class="btn btn-primary" onclick="verifyOTP()"
                id="verify-btn" style="display:none">{t['verify']}</button>
      </div>

      <!-- Email/password tab -->
      <div id="pane-email" {'style="display:none"' if tab=='phone' else ''}>
        <form method="post" action="/apply/login-email">
          <div class="form-group">
            <label class="lbl">{t['email']}</label>
            <input type="email" name="email" placeholder="you@email.com" required>
          </div>
          <div class="form-group">
            <label class="lbl">{t['password']}</label>
            <input type="password" name="password" required>
          </div>
          <button type="submit" class="btn btn-primary">{t['login']}</button>
        </form>
        <hr class="divider">
        <p style="text-align:center;font-size:13px;color:var(--muted);margin-bottom:10px">
          New applicant? <a href="/apply/register" style="color:var(--primary);font-weight:600">{t['register']}</a>
        </p>
      </div>
    </div>

    <script>
    function showTab(tab) {{
      document.getElementById('pane-phone').style.display = tab==='phone' ? '' : 'none';
      document.getElementById('pane-email').style.display = tab==='email' ? '' : 'none';
      document.getElementById('tab-phone').className = 'tab' + (tab==='phone' ? ' active' : '');
      document.getElementById('tab-email').className = 'tab' + (tab==='email' ? ' active' : '');
    }}

    async function sendOTP() {{
      const phone = document.getElementById('phone-input').value.trim();
      if (!phone) return;
      const msg = document.getElementById('otp-msg');
      msg.innerHTML = '<div class="alert alert-info"><span>ℹ</span><span>Sending...</span></div>';
      const r = await fetch('/apply/send-otp', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
        body: 'phone=' + encodeURIComponent(phone)
      }});
      const d = await r.json();
      if (d.sent) {{
        document.getElementById('otp-box').style.display = '';
        document.getElementById('verify-btn').style.display = '';
        msg.innerHTML = '<div class="alert alert-ok"><span>✓</span><span>' + d.message + '</span></div>';
      }} else {{
        msg.innerHTML = '<div class="alert alert-err"><span>✕</span><span>' + d.message + '</span></div>';
      }}
    }}

    async function verifyOTP() {{
      const phone = document.getElementById('phone-input').value.trim();
      const otp   = document.getElementById('otp-input').value.trim();
      const r = await fetch('/apply/verify-otp', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
        body: 'phone=' + encodeURIComponent(phone) + '&otp=' + encodeURIComponent(otp)
      }});
      const d = await r.json();
      if (d.success) {{
        window.location.href = d.redirect;
      }} else {{
        document.getElementById('otp-msg').innerHTML =
          '<div class="alert alert-err"><span>✕</span><span>' + d.message + '</span></div>';
      }}
    }}
    </script>"""

    return HTMLResponse(_shell(request, content, alert=err))


@app.post("/send-otp")
async def send_otp(request: Request, phone: str = Form("")):
    phone = re.sub(r"[\s\-()]", "", phone)
    if not phone:
        return {"sent": False, "message": "Please enter your phone number"}

    otp     = _gen_otp()
    expires = datetime.utcnow() + timedelta(minutes=10)

    # Upsert applicant record
    db = SessionLocal()
    try:
        existing = db.execute(text("SELECT id FROM applicants WHERE phone=:p"), {"p": phone}).fetchone()
        if existing:
            db.execute(text("UPDATE applicants SET otp_code=:o, otp_expires=:e WHERE phone=:p"),
                       {"o": otp, "e": expires, "p": phone})
        else:
            db.execute(text("INSERT INTO applicants (phone, otp_code, otp_expires) VALUES (:p,:o,:e)"),
                       {"p": phone, "o": otp, "e": expires})
        db.commit()
    except Exception as e:
        db.rollback()
        return {"sent": False, "message": "Error — please try again"}
    finally:
        db.close()

    _send_otp(phone, otp)

    # In dev mode show OTP in response for testing
    dev_mode = os.getenv("DEV_MODE", "true").lower() == "true"
    msg = f"Code sent to {phone}." + (f" [DEV: {otp}]" if dev_mode else "")
    return {"sent": True, "message": msg}


@app.post("/verify-otp")
async def verify_otp(request: Request, phone: str = Form(""), otp: str = Form("")):
    phone = re.sub(r"[\s\-()]", "", phone)
    db    = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT id, name, otp_code, otp_expires FROM applicants WHERE phone=:p"
        ), {"p": phone}).fetchone()

        if not row:
            return {"success": False, "message": "Phone number not found"}
        if row.otp_code != otp:
            return {"success": False, "message": "Incorrect code — please try again"}
        if datetime.utcnow() > row.otp_expires:
            return {"success": False, "message": "Code expired — please request a new one"}

        db.execute(text("UPDATE applicants SET verified=TRUE, otp_code=NULL WHERE id=:id"),
                   {"id": row.id})
        db.commit()

        token = _make_session(row.id, row.name or phone)
        resp  = {"success": True, "redirect": "/apply/"}
        # Return token via cookie — set in JS using a redirect
        return {"success": True, "redirect": f"/apply/otp-login?id={row.id}&token={token}"}
    finally:
        db.close()


@app.get("/otp-login")
async def otp_login(id: int, token: str):
    r = RedirectResponse("/apply/", 303)
    r.set_cookie("applicant_session", token, httponly=True, samesite="lax", max_age=24*3600)
    return r


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, error: str = ""):
    t    = T.get(_lang(request), T["en"])
    err  = _alert(error, "err") if error else ""
    content = f"""
    <div class="hero" style="padding:20px 20px 10px">
      <h1>{t['register']}</h1>
    </div>
    <div class="card">
      <form method="post" action="/apply/register">
        <div class="form-group">
          <label class="lbl">{t['name']} *</label>
          <input type="text" name="name" required placeholder="Your full legal name">
        </div>
        <div class="grid-2">
          <div class="form-group">
            <label class="lbl">{t['email']} *</label>
            <input type="email" name="email" required>
          </div>
          <div class="form-group">
            <label class="lbl">{t['phone']}</label>
            <input type="tel" name="phone" placeholder="+855 xx xxx xxx">
          </div>
        </div>
        <div class="form-group">
          <label class="lbl">{t['password']} *</label>
          <input type="password" name="password" required minlength="8"
                 placeholder="At least 8 characters">
        </div>
        <div class="form-group">
          <label class="lbl">{t['dob']}</label>
          <input type="date" name="dob">
        </div>
        <button type="submit" class="btn btn-primary">{t['register']}</button>
        <p style="text-align:center;font-size:13px;color:var(--muted);margin-top:12px">
          <a href="/apply/login" style="color:var(--primary)">{t['login']}</a>
        </p>
      </form>
    </div>"""
    return HTMLResponse(_shell(request, content, alert=err))


@app.post("/register")
async def register(request: Request,
    name: str=Form(""), email: str=Form(""), phone: str=Form(""),
    password: str=Form(""), dob: str=Form("")):
    db = SessionLocal()
    try:
        db.execute(text("""
            INSERT INTO applicants (name, email, phone, password, dob, verified)
            VALUES (:n,:e,:p,:pw,:d,TRUE)
        """), {"n":name,"e":email,"p":phone or None,"pw":_hash(password),"d":dob or None})
        db.commit()
        row = db.execute(text("SELECT id FROM applicants WHERE email=:e"),{"e":email}).fetchone()
        token = _make_session(row.id, name)
        resp  = RedirectResponse("/apply/", 303)
        resp.set_cookie("applicant_session", token, httponly=True, samesite="lax", max_age=24*3600)
        return resp
    except Exception as e:
        db.rollback()
        return RedirectResponse(f"/apply/register?error=Email+already+registered", 303)
    finally:
        db.close()


@app.post("/login-email")
async def login_email(request: Request, email: str=Form(""), password: str=Form("")):
    db = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT id, name FROM applicants WHERE email=:e AND password=:p"
        ), {"e":email,"p":_hash(password)}).fetchone()
        if not row:
            return RedirectResponse("/apply/login?tab=email&error=Invalid+email+or+password", 303)
        token = _make_session(row.id, row.name or email)
        resp  = RedirectResponse("/apply/", 303)
        resp.set_cookie("applicant_session", token, httponly=True, samesite="lax", max_age=24*3600)
        return resp
    finally:
        db.close()


@app.get("/logout")
def logout():
    r = RedirectResponse("/apply/login", 303)
    r.delete_cookie("applicant_session")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    user  = _guard(request)
    t     = T.get(_lang(request), T["en"])
    lang  = _lang(request)
    loans = _q("SELECT * FROM applicant_loans WHERE applicant_id=:id ORDER BY created_at DESC",
               {"id": user["id"]})

    cards = ""
    for l in loans:
        status = l.get("status","draft")
        amt    = f"${float(l.get('loan_amount',0)):,.0f}"
        dt     = str(l.get("created_at",""))[:10]
        ltype  = l.get("loan_type","")
        ref    = l.get("loan_ref","")
        cards += f"""
        <a href="/apply/case/{ref}" style="text-decoration:none">
          <div class="app-card">
            <div>
              <div class="app-ref">{ref}</div>
              <div class="app-meta">{ltype} · {amt} · {dt}</div>
            </div>
            {_status_pill(status, lang)}
          </div>
        </a>"""

    if not cards:
        cards = f"""
        <div style="text-align:center;padding:40px;color:var(--muted)">
          <div style="font-size:40px;margin-bottom:12px">📄</div>
          <p>{t['no_apps']}</p>
          <a href="/apply/new" class="btn btn-primary" style="width:auto;margin-top:16px;padding:10px 24px">{t['new_app']}</a>
        </div>"""

    content = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
      <div>
        <h2 style="font-size:18px;font-weight:700">{t['welcome']}, {user.get('name','')}</h2>
        <p style="font-size:13px;color:var(--muted);margin-top:2px">{t['my_apps']}</p>
      </div>
      <a href="/apply/new" class="btn btn-primary" style="width:auto;padding:9px 18px">{t['new_app']}</a>
    </div>
    {cards}"""

    return HTMLResponse(_shell(request, content, user=user))


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/new", response_class=HTMLResponse)
def new_application(request: Request):
    user = _guard(request)
    t    = T.get(_lang(request), T["en"])
    lang = _lang(request)

    # Pre-fill applicant info
    info = _q("SELECT * FROM applicants WHERE id=:id", {"id": user["id"]})
    a    = info[0] if info else {}

    loan_opts = "".join(f"<option>{lt}</option>" for lt in LOAN_TYPES.get(lang, LOAN_TYPES["en"]))

    content = f"""
    <h2 style="font-size:18px;font-weight:700;margin-bottom:16px">{t['new_app']}</h2>

    <form action="/apply/new" method="post" enctype="multipart/form-data">

      <div class="card">
        <div class="card-title">{t['personal_info']}</div>
        <div class="form-group">
          <label class="lbl">{t['name']} *</label>
          <input type="text" name="name" value="{a.get('name','')}" required>
        </div>
        <div class="grid-2">
          <div class="form-group">
            <label class="lbl">{t['dob']}</label>
            <input type="date" name="dob" value="{a.get('dob','') or ''}">
          </div>
          <div class="form-group">
            <label class="lbl">{t['income']}</label>
            <input type="number" name="income" min="0" step="10"
                   value="{int(a.get('income',0) or 0)}">
          </div>
        </div>
        <div class="form-group">
          <label class="lbl">{t['employer']}</label>
          <input type="text" name="employer" value="{a.get('employer','') or ''}">
        </div>
        <div class="form-group">
          <label class="lbl">{t['address']}</label>
          <textarea name="address" rows="2">{a.get('address','') or ''}</textarea>
        </div>
      </div>

      <div class="card">
        <div class="card-title">{t['loan_details']}</div>
        <div class="form-group">
          <label class="lbl">{t['loan_type']} *</label>
          <select name="loan_type" required>{loan_opts}</select>
        </div>
        <div class="grid-2">
          <div class="form-group">
            <label class="lbl">{t['loan_amount']} *</label>
            <input type="number" name="loan_amount" min="100" step="100" required placeholder="5000">
          </div>
          <div class="form-group">
            <label class="lbl">{t['loan_term']} *</label>
            <input type="number" name="loan_term" min="1" max="360" value="12" required>
          </div>
        </div>
        <div class="form-group">
          <label class="lbl">{t['loan_purpose']} *</label>
          <input type="text" name="loan_purpose" required placeholder="e.g. Working capital">
        </div>
      </div>

      <div class="card">
        <div class="card-title">{t['upload_docs']}</div>
        <div class="alert alert-info">
          <span>ℹ</span><span>{t['docs_help']}</span>
        </div>
        <div class="form-group">
          <label class="lbl">{t['id_card']} *</label>
          <input type="file" name="id_card" accept=".jpg,.jpeg,.png" required>
        </div>
        <div class="form-group">
          <label class="lbl">{t['selfie']}</label>
          <input type="file" name="selfie" accept=".jpg,.jpeg,.png">
        </div>
        <div class="form-group">
          <label class="lbl">{t['bank_statement']}</label>
          <input type="file" name="bank_statement" accept=".pdf,.jpg,.jpeg,.png">
        </div>
        <div class="form-group">
          <label class="lbl">{t['payslip']}</label>
          <input type="file" name="payslip" accept=".pdf,.jpg,.jpeg,.png">
        </div>
        <div class="form-group">
          <label class="lbl">{t['utility_bill']}</label>
          <input type="file" name="utility_bill" accept=".pdf,.jpg,.jpeg,.png">
        </div>
      </div>

      <button type="submit" name="action" value="submit" class="btn btn-primary">{t['submit']}</button>
      <button type="submit" name="action" value="draft"
              class="btn btn-ghost" style="margin-top:8px">{t['save_draft']}</button>
    </form>"""

    return HTMLResponse(_shell(request, content, user=user))


@app.post("/new")
async def submit_application(
    request: Request,
    action:       str = Form("submit"),
    name:         str = Form(""),
    dob:          str = Form(""),
    income:       str = Form("0"),
    employer:     str = Form(""),
    address:      str = Form(""),
    loan_type:    str = Form(""),
    loan_amount:  str = Form("0"),
    loan_term:    str = Form("12"),
    loan_purpose: str = Form(""),
    id_card:      UploadFile = File(None),
    selfie:       UploadFile = File(None),
    bank_statement: UploadFile = File(None),
    payslip:      UploadFile = File(None),
    utility_bill: UploadFile = File(None),
):
    user = _guard(request)

    # Update applicant profile
    _exec("""UPDATE applicants SET name=:n, dob=:d, income=:i, employer=:e, address=:a
             WHERE id=:id""",
          {"n":name,"d":dob or None,"i":float(income or 0),"e":employer,"a":address,"id":user["id"]})

    # Create loan record
    loan_ref = _gen_ref()
    _exec("""INSERT INTO applicant_loans
             (loan_ref, applicant_id, loan_type, loan_amount, loan_term, loan_purpose, status)
             VALUES (:ref,:aid,:lt,:la,:lterm,:lp,:status)""",
          {"ref":loan_ref,"aid":user["id"],"lt":loan_type,
           "la":float(loan_amount or 0),"lterm":int(loan_term or 12),
           "lp":loan_purpose,"status":"draft"})

    if action == "draft":
        return RedirectResponse(f"/apply/case/{loan_ref}", 303)

    # Submit to KYC API
    try:
        files        = {}
        docs_uploaded = []
        for fname, upload in [("id_card",id_card),("selfie",selfie),
                               ("bank_statement",bank_statement),
                               ("payslip",payslip),("utility_bill",utility_bill)]:
            if upload and upload.filename:
                data = await upload.read()
                if data:
                    files[fname] = (upload.filename, data, upload.content_type or "application/octet-stream")
                    docs_uploaded.append(upload.filename)

        _exec("UPDATE applicant_loans SET status='screening', submitted_at=NOW(), docs_uploaded=:d WHERE loan_ref=:r",
              {"d":json.dumps(docs_uploaded),"r":loan_ref})

        async with httpx.AsyncClient(base_url=KYC_API_URL, timeout=120) as client:
            r = await client.post("/api/v1/loan-case",
                data={
                    "loan_ref":      loan_ref,
                    "applicant_id":  str(user["id"]),
                    "applicant_name": name,
                    "loan_type":     loan_type,
                    "loan_amount":   loan_amount,
                    "source_system": "ApplicantPortal",
                    "callback_url":  f"http://applicant-portal:8006/webhook/kyc",
                },
                files=files or {"_x":("x.txt",b"x","text/plain")},
            )

        if r.status_code == 200:
            res     = r.json()
            overall = res.get("overall",{})
            _exec("""UPDATE applicant_loans SET
                     kyc_status='complete', kyc_risk_level=:rl, kyc_risk_score=:rs,
                     kyc_action=:ra, kyc_flags=:rf, kyc_result=:rr, status='review'
                     WHERE loan_ref=:ref""",
                  {"rl":overall.get("risk_level"),"rs":overall.get("risk_score"),
                   "ra":overall.get("action"),"rf":json.dumps(overall.get("flags",[])),
                   "rr":json.dumps(res),"ref":loan_ref})
        else:
            _exec("UPDATE applicant_loans SET kyc_status='failed' WHERE loan_ref=:r",{"r":loan_ref})
    except Exception as e:
        log.error(f"KYC submit failed: {e}")
        _exec("UPDATE applicant_loans SET kyc_status='failed' WHERE loan_ref=:r",{"r":loan_ref})

    return RedirectResponse(f"/apply/case/{loan_ref}", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE STATUS VIEW
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/case/{loan_ref}", response_class=HTMLResponse)
def case_status(loan_ref: str, request: Request):
    user = _guard(request)
    t    = T.get(_lang(request), T["en"])
    lang = _lang(request)

    rows = _q("SELECT * FROM applicant_loans WHERE loan_ref=:r AND applicant_id=:id",
              {"r":loan_ref,"id":user["id"]})
    if not rows:
        raise HTTPException(404)
    loan   = rows[0]
    status = loan.get("status","draft")

    steps_html = ""
    step_keys  = ["submitted","screening","review","approved"]
    step_done  = {"draft":0,"submitted":1,"screening":2,"review":3,"approved":4,"rejected":3}.get(status,0)
    for i, sk in enumerate(step_keys):
        cls = "step done" if i < step_done else ("step active" if i == step_done-1 else "step")
        lbl = t.get(sk, sk)
        steps_html += f'<div class="{cls}"><div class="step-dot">{"✓" if i<step_done else str(i+1)}</div><div class="step-label">{lbl}</div></div>'

    kyc_flags = loan.get("kyc_flags") or []
    if isinstance(kyc_flags, str):
        try: kyc_flags = json.loads(kyc_flags)
        except: kyc_flags = []

    # Applicant-friendly status message
    status_msgs = {
        "en": {
            "draft":     "Your application is saved as a draft. Submit when ready.",
            "submitted": "Your application has been received. We are processing your documents.",
            "screening": "Our team is reviewing your documents. This usually takes 1-2 business days.",
            "review":    "Your application is under final review by our loan officers.",
            "approved":  "Congratulations! Your loan application has been approved.",
            "rejected":  "We were unable to approve your application at this time.",
        },
        "km": {
            "draft":     "ពាក្យសុំរបស់អ្នកត្រូវបានរក្សាទុកជាព្រាង។ ដាក់ស្នើពេលរួចរាល់។",
            "submitted": "ពាក្យសុំរបស់អ្នកត្រូវបានទទួល។ យើងកំពុងដំណើរការឯកសាររបស់អ្នក។",
            "screening": "ក្រុមការងាររបស់យើងកំពុងពិនិត្យឯកសាររបស់អ្នក។ ជាធម្មតាចំណាយពេល ១-២ ថ្ងៃធ្វើការ។",
            "review":    "ពាក្យសុំរបស់អ្នកស្ថិតនៅក្រោមការពិនិត្យចុងក្រោយ។",
            "approved":  "សូមអបអរ! ពាក្យសុំប្រាក់កម្ចីរបស់អ្នកត្រូវបានអនុម័ត។",
            "rejected":  "យើងមិនអាចអនុម័តពាក្យសុំរបស់អ្នកនាពេលនេះទេ។",
        }
    }
    msg = status_msgs.get(lang,status_msgs["en"]).get(status,"")

    msg_kind = {"approved":"ok","rejected":"err","review":"warn"}.get(status,"info")
    status_alert = _alert(msg, msg_kind) if msg else ""

    decision_note = ""
    if loan.get("decision_notes") and status in ("approved","rejected"):
        decision_note = f"""
        <div class="card">
          <div class="card-title">{t['decision_notes']}</div>
          <p style="font-size:14px;line-height:1.6">{loan['decision_notes']}</p>
        </div>"""

    content = f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
      <a href="/apply/" style="color:var(--muted);font-size:13px">← {t['my_apps']}</a>
    </div>

    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div>
          <div style="font-size:11px;color:var(--muted);margin-bottom:3px">{t['ref']}</div>
          <div style="font-family:monospace;font-size:18px;font-weight:700;color:var(--primary)">{loan_ref}</div>
        </div>
        {_status_pill(status, lang)}
      </div>

      <div class="step-row">{steps_html}</div>
      {_progress(status)}
      {status_alert}

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px">
        <div>
          <div style="color:var(--muted);margin-bottom:2px">{t['type_label']}</div>
          <div style="font-weight:600">{loan.get('loan_type','—')}</div>
        </div>
        <div>
          <div style="color:var(--muted);margin-bottom:2px">{t['amount_label']}</div>
          <div style="font-weight:700;font-size:16px">${float(loan.get('loan_amount',0)):,.0f}</div>
        </div>
        <div>
          <div style="color:var(--muted);margin-bottom:2px">{t['loan_term']}</div>
          <div>{loan.get('loan_term','—')} months</div>
        </div>
        <div>
          <div style="color:var(--muted);margin-bottom:2px">{t['date_label']}</div>
          <div>{str(loan.get('created_at','—'))[:10]}</div>
        </div>
      </div>
    </div>

    {decision_note}

    <div class="card">
      <div class="card-title">{t['upload_docs']}</div>
      {''.join(f"<div style='padding:6px 0;border-bottom:1px solid #fef2f2;font-size:13px'>📄 {d}</div>" for d in (json.loads(loan['docs_uploaded']) if loan.get('docs_uploaded') else [])) or "<p style='color:var(--muted);font-size:13px'>No documents uploaded yet</p>"}
    </div>"""

    return HTMLResponse(_shell(request, content, user=user))


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBHOOK — receive KYC results
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/kyc")
async def kyc_webhook(request: Request):
    try:
        data     = await request.json()
        loan_ref = data.get("loan_ref")
        if not loan_ref: return {"received": True}
        overall  = data.get("overall", {})
        _exec("""UPDATE applicant_loans SET
                 kyc_status='complete', kyc_risk_level=:rl, kyc_risk_score=:rs,
                 kyc_action=:ra, kyc_result=:rr, status='review'
                 WHERE loan_ref=:ref""",
              {"rl":overall.get("risk_level"),"rs":overall.get("risk_score"),
               "ra":overall.get("action"),"rr":json.dumps(data),"ref":loan_ref})
    except Exception as e:
        log.error(f"Webhook error: {e}")
    return {"received": True}


@app.get("/health")
def health():
    return {"status":"running","service":"applicant-portal","version":"1.0.0"}
