"""
portal.py — Internal Bank Staff Portal
Full web UI for document submission, case management, and compliance reporting.
Mounted at /portal/ via Nginx.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("fraud_detect.portal")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI(title="Staff Portal", version="1.0.0")

RISK_COLOR  = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
ACTION_COLOR = {"REJECT": "#ef4444", "REVIEW": "#f59e0b", "PASS": "#22c55e"}


def _q(sql, params={}):
    db = SessionLocal()
    try:
        return [dict(r._mapping) for r in db.execute(text(sql), params)]
    except Exception as e:
        log.error(f"Query error: {e}")
        return []
    finally:
        db.close()


def _iso(dt):
    return dt.isoformat() if dt else ""


def _badge(level, text_override=None):
    action = text_override or {"HIGH":"REJECT","MEDIUM":"REVIEW","LOW":"PASS"}.get(level, level)
    color  = ACTION_COLOR.get(action, RISK_COLOR.get(level, "#94a3b8"))
    return f'<span style="background:{color};color:#fff;padding:3px 12px;border-radius:4px;font-size:11px;font-weight:700">{action}</span>'


def _nav(active=""):
    pages = [
        ("dashboard", "📊 Dashboard", "/portal/"),
        ("submit",    "📤 Submit",    "/portal/submit"),
        ("cases",     "📁 Cases",     "/portal/cases"),
        ("batch",     "📦 Batch",     "/portal/batch"),
        ("expiry",    "⏰ Expiry",    "/portal/expiry"),
    ]
    links = ""
    for key, label, href in pages:
        active_style = "background:#1e40af;color:#fff;" if key == active else "color:#cbd5e1;"
        links += f'<a href="{href}" style="display:block;padding:10px 16px;border-radius:6px;text-decoration:none;font-size:13px;font-weight:500;{active_style}margin-bottom:2px">{label}</a>'
    return f'''
    <div style="width:200px;min-height:100vh;background:#0f172a;padding:20px 12px;flex-shrink:0">
      <div style="color:#fff;font-size:15px;font-weight:700;padding:8px 16px;margin-bottom:20px">
        🔍 KYC Portal
      </div>
      {links}
    </div>'''


def _shell(title, content, active=""):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title} — KYC Portal</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#f8fafc;color:#0f172a;display:flex}}
    .main{{flex:1;padding:28px 32px;overflow-x:hidden}}
    h2{{font-size:20px;font-weight:700;margin-bottom:20px;color:#0f172a}}
    .card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:24px;margin-bottom:20px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th{{padding:10px 12px;text-align:left;font-size:11px;font-weight:600;color:#64748b;
        border-bottom:2px solid #f1f5f9;text-transform:uppercase;letter-spacing:.4px}}
    td{{padding:10px 12px;border-bottom:1px solid #f8fafc;vertical-align:middle}}
    tr:hover td{{background:#f8fafc}}
    input,select,textarea{{width:100%;padding:9px 12px;border:1px solid #e2e8f0;
      border-radius:6px;font-size:13px;outline:none;font-family:inherit}}
    input:focus,select:focus{{border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.1)}}
    .btn{{padding:9px 20px;background:#3b82f6;color:#fff;border:none;border-radius:6px;
          cursor:pointer;font-size:13px;font-weight:500;text-decoration:none;display:inline-block}}
    .btn:hover{{background:#2563eb}}
    .btn-red{{background:#ef4444}}.btn-red:hover{{background:#dc2626}}
    .btn-green{{background:#22c55e}}.btn-green:hover{{background:#16a34a}}
    .btn-gray{{background:#64748b}}.btn-gray:hover{{background:#475569}}
    label{{font-size:12px;font-weight:600;color:#374151;display:block;margin-bottom:5px}}
    .form-group{{margin-bottom:16px}}
    .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
    .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
    .stat{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px 22px}}
    .stat .num{{font-size:26px;font-weight:700}}
    .stat .lbl{{font-size:12px;color:#64748b;margin-top:3px}}
    .alert{{padding:12px 16px;border-radius:6px;font-size:13px;margin-bottom:16px}}
    .alert-red{{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}}
    .alert-green{{background:#f0fdf4;border:1px solid #bbf7d0;color:#166534}}
    .alert-yellow{{background:#fffbeb;border:1px solid #fde68a;color:#92400e}}
    .tag{{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;
          border-radius:3px;padding:2px 7px;font-size:11px;margin:2px;color:#475569}}
  </style>
</head>
<body>
  {_nav(active)}
  <div class="main">
    <h2>{title}</h2>
    {content}
  </div>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def dashboard():
    since30 = datetime.utcnow() - timedelta(days=30)
    since7  = datetime.utcnow() - timedelta(days=7)

    stats30 = _q("""
        SELECT COUNT(*) total,
               COUNT(*) FILTER (WHERE risk_level='HIGH')   high,
               COUNT(*) FILTER (WHERE risk_level='MEDIUM') medium,
               COUNT(*) FILTER (WHERE risk_level='LOW')    low,
               COUNT(*) FILTER (WHERE doc_type='id_card')  id_cards,
               COUNT(*) FILTER (WHERE doc_type='pdf')      pdfs,
               ROUND(AVG(risk_score),1) avg_score,
               COUNT(DISTINCT applicant_id)
                 FILTER (WHERE applicant_id IS NOT NULL)   applicants
        FROM screening_logs WHERE screened_at >= :s
    """, {"s": since30})
    s = stats30[0] if stats30 else {}

    # expiry warnings
    expiring = _q("""
        SELECT COUNT(*) cnt FROM screening_logs
        WHERE doc_type='id_card' AND screened_at >= :s
        AND full_result::text LIKE '%id_card_expired%'
    """, {"s": since30})
    exp_count = expiring[0]["cnt"] if expiring else 0

    # recent HIGH risk
    recent_high = _q("""
        SELECT file_name, applicant_id, risk_score, screened_at, id_number
        FROM screening_logs
        WHERE risk_level='HIGH' AND screened_at >= :s
        ORDER BY screened_at DESC LIMIT 5
    """, {"s": since7})

    high_rows = ""
    for r in recent_high:
        dt = str(r.get("screened_at",""))[:16].replace("T"," ")
        high_rows += f"""<tr>
          <td>{dt}</td>
          <td>{r.get("file_name","")[:30]}</td>
          <td>{r.get("applicant_id") or "—"}</td>
          <td>{r.get("id_number") or "—"}</td>
          <td style="font-weight:700;color:#ef4444">{r.get("risk_score","")}</td>
          <td><a href="/portal/cases?applicant={r.get('applicant_id','')}"
                 class="btn" style="padding:4px 10px;font-size:11px">View</a></td>
        </tr>"""

    content = f"""
    <div class="grid-3" style="margin-bottom:20px">
      <div class="stat"><div class="num">{s.get("total",0)}</div><div class="lbl">Total Screened (30d)</div></div>
      <div class="stat"><div class="num" style="color:#ef4444">{s.get("high",0)}</div><div class="lbl">HIGH Risk</div></div>
      <div class="stat"><div class="num" style="color:#f59e0b">{s.get("medium",0)}</div><div class="lbl">MEDIUM Risk</div></div>
      <div class="stat"><div class="num" style="color:#22c55e">{s.get("low",0)}</div><div class="lbl">LOW Risk</div></div>
      <div class="stat"><div class="num">{s.get("applicants",0)}</div><div class="lbl">Unique Applicants</div></div>
      <div class="stat"><div class="num" style="color:#f59e0b">{exp_count}</div><div class="lbl">Expired IDs Detected</div></div>
    </div>

    <div class="card">
      <h3 style="font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
                 letter-spacing:.4px;margin-bottom:14px">⚠ Recent HIGH Risk Submissions (7 days)</h3>
      {"<p style='color:#94a3b8;font-size:13px'>No HIGH risk submissions in the last 7 days</p>" if not recent_high else f"""
      <table>
        <thead><tr><th>Date</th><th>File</th><th>Applicant</th><th>ID Number</th><th>Score</th><th></th></tr></thead>
        <tbody>{high_rows}</tbody>
      </table>"""}
    </div>

    <div class="grid-2">
      <div class="card" style="text-align:center">
        <div style="font-size:36px;margin-bottom:8px">📤</div>
        <div style="font-weight:600;margin-bottom:8px">Submit Document</div>
        <div style="color:#64748b;font-size:13px;margin-bottom:14px">Screen a single document</div>
        <a href="/portal/submit" class="btn">Submit Now</a>
      </div>
      <div class="card" style="text-align:center">
        <div style="font-size:36px;margin-bottom:8px">📦</div>
        <div style="font-weight:600;margin-bottom:8px">Batch Processing</div>
        <div style="color:#64748b;font-size:13px;margin-bottom:14px">Upload multiple documents via CSV</div>
        <a href="/portal/batch" class="btn">Start Batch</a>
      </div>
    </div>"""

    return HTMLResponse(_shell("Dashboard", content, "dashboard"))


# ═══════════════════════════════════════════════════════════════════════════════
#  SUBMIT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/submit", response_class=HTMLResponse)
def submit_page(success: str = "", error: str = ""):
    msg = ""
    if success:
        msg = f'<div class="alert alert-green">✅ Document submitted successfully. Task ID: <strong>{success}</strong></div>'
    if error:
        msg = f'<div class="alert alert-red">❌ Error: {error}</div>'

    content = f"""
    {msg}
    <div class="grid-2">
      <div class="card">
        <h3 style="font-size:14px;font-weight:600;margin-bottom:16px">📄 Screen PDF Document</h3>
        <form action="/portal/submit/pdf" method="post" enctype="multipart/form-data">
          <div class="form-group">
            <label>Applicant ID</label>
            <input type="text" name="applicant_id" placeholder="e.g. APP-2024-001">
          </div>
          <div class="form-group">
            <label>PDF File *</label>
            <input type="file" name="file" accept=".pdf" required>
          </div>
          <button type="submit" class="btn">Screen PDF</button>
        </form>
      </div>

      <div class="card">
        <h3 style="font-size:14px;font-weight:600;margin-bottom:16px">🪪 Screen ID Card</h3>
        <form action="/portal/submit/idcard" method="post" enctype="multipart/form-data">
          <div class="form-group">
            <label>Applicant ID</label>
            <input type="text" name="applicant_id" placeholder="e.g. APP-2024-001">
          </div>
          <div class="form-group">
            <label>ID Card Image *</label>
            <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
          </div>
          <div class="form-group">
            <label>Selfie (optional — for face match)</label>
            <input type="file" name="selfie" accept=".jpg,.jpeg,.png">
          </div>
          <div class="form-group">
            <label>Callback URL (optional)</label>
            <input type="text" name="callback_url"
                   placeholder="http://your-system/webhook" value="http://172.16.26.48:3000/webhook/kyc">
          </div>
          <button type="submit" class="btn">Screen ID Card</button>
        </form>
      </div>

      <div class="card">
        <h3 style="font-size:14px;font-weight:600;margin-bottom:16px">🖼 Screen Document Image</h3>
        <form action="/portal/submit/image" method="post" enctype="multipart/form-data">
          <div class="form-group">
            <label>Applicant ID</label>
            <input type="text" name="applicant_id" placeholder="e.g. APP-2024-001">
          </div>
          <div class="form-group">
            <label>Image File *</label>
            <input type="file" name="file" accept=".jpg,.jpeg,.png,.bmp,.tiff" required>
          </div>
          <button type="submit" class="btn">Screen Image</button>
        </form>
      </div>

      <div class="card">
        <h3 style="font-size:14px;font-weight:600;margin-bottom:16px">🔍 Poll Task Result</h3>
        <div class="form-group">
          <label>Task ID</label>
          <input type="text" id="task-id-input" placeholder="Paste task_id here...">
        </div>
        <button class="btn" onclick="pollResult()">Check Result</button>
        <div id="poll-result" style="margin-top:14px"></div>
        <script>
        async function pollResult() {{
          const tid = document.getElementById('task-id-input').value.trim();
          if (!tid) return;
          const div = document.getElementById('poll-result');
          div.innerHTML = '<p style="color:#64748b;font-size:13px">Checking...</p>';
          try {{
            const r = await fetch('/result/' + tid);
            const d = await r.json();
            const status = d.status;
            if (status === 'complete') {{
              const risk = d.result?.risk || {{}};
              const color = risk.level === 'HIGH' ? '#ef4444' : risk.level === 'MEDIUM' ? '#f59e0b' : '#22c55e';
              div.innerHTML = `<div class="alert alert-green">
                ✅ Complete — Risk: <strong style="color:${{color}}">${{risk.level}}</strong>
                Score: ${{risk.score}} — Action: <strong>${{risk.action}}</strong><br>
                <a href="/portal/cases?task=${{tid}}" class="btn" style="margin-top:8px;padding:4px 12px;font-size:11px">View Full Report</a>
              </div>`;
            }} else if (status === 'failed') {{
              div.innerHTML = `<div class="alert alert-red">❌ Failed: ${{d.error}}</div>`;
            }} else {{
              div.innerHTML = `<div class="alert alert-yellow">⏳ Status: ${{status}} — Try again in 10 seconds</div>`;
              setTimeout(pollResult, 10000);
            }}
          }} catch(e) {{
            div.innerHTML = '<div class="alert alert-red">Error fetching result</div>';
          }}
        }}
        </script>
      </div>
    </div>"""

    return HTMLResponse(_shell("Submit Document", content, "submit"))


@app.post("/submit/pdf")
async def submit_pdf(request: Request, applicant_id: str = Form(""), file: UploadFile = File(...)):
    import httpx
    async with httpx.AsyncClient(base_url="http://api:8001") as client:
        resp = await client.post("/screen-pdf",
            files={"file": (file.filename, await file.read(), file.content_type)},
            data={"applicant_id": applicant_id}, timeout=60)
    if resp.status_code == 200:
        result = resp.json()
        risk = result.get("risk", {})
        return RedirectResponse(f"/portal/cases?msg=PDF+screened+Risk:{risk.get('level','?')}", 303)
    return RedirectResponse(f"/portal/submit?error=PDF+screening+failed", 303)


@app.post("/submit/image")
async def submit_image(request: Request, applicant_id: str = Form(""), file: UploadFile = File(...)):
    import httpx
    async with httpx.AsyncClient(base_url="http://api:8001") as client:
        resp = await client.post("/screen-image",
            files={"file": (file.filename, await file.read(), file.content_type)},
            data={"applicant_id": applicant_id}, timeout=60)
    if resp.status_code == 200:
        return RedirectResponse(f"/portal/cases?msg=Image+screened+successfully", 303)
    return RedirectResponse(f"/portal/submit?error=Image+screening+failed", 303)


@app.post("/submit/idcard")
async def submit_idcard(
    request: Request,
    applicant_id: str = Form(""),
    callback_url: str = Form(""),
    file:   UploadFile = File(...),
    selfie: UploadFile = File(None),
):
    import httpx
    files = {"file": (file.filename, await file.read(), file.content_type)}
    if selfie and selfie.filename:
        files["selfie"] = (selfie.filename, await selfie.read(), selfie.content_type)
    data = {"applicant_id": applicant_id}
    if callback_url:
        data["callback_url"] = callback_url
    async with httpx.AsyncClient(base_url="http://api:8001") as client:
        resp = await client.post("/screen-id-card", files=files, data=data, timeout=60)
    if resp.status_code == 202:
        task_id = resp.json().get("task_id", "")
        return RedirectResponse(f"/portal/submit?success={task_id}", 303)
    return RedirectResponse(f"/portal/submit?error=ID+card+submission+failed", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  CASES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/cases", response_class=HTMLResponse)
def cases_page(
    applicant: str = "",
    risk: str = "",
    doc_type: str = "",
    days: int = 30,
    msg: str = "",
):
    since = datetime.utcnow() - timedelta(days=days)

    filters = "WHERE screened_at >= :since"
    params  = {"since": since}
    if applicant:
        filters += " AND applicant_id ILIKE :app"
        params["app"] = f"%{applicant}%"
    if risk:
        filters += " AND risk_level = :risk"
        params["risk"] = risk
    if doc_type:
        filters += " AND doc_type = :dt"
        params["dt"] = doc_type

    rows = _q(f"""
        SELECT id, file_name, doc_type, risk_level, risk_score,
               flags, screened_at, applicant_id, id_number
        FROM screening_logs {filters}
        ORDER BY screened_at DESC LIMIT 50
    """, params)

    alert = f'<div class="alert alert-green">✅ {msg}</div>' if msg else ""

    table_rows = ""
    for r in rows:
        level    = r.get("risk_level", "—")
        score    = r.get("risk_score", "—")
        fname    = r.get("file_name", "")
        dtype    = r.get("doc_type", "").upper()
        applicant_id = r.get("applicant_id") or "—"
        id_num   = r.get("id_number") or "—"
        dt       = str(r.get("screened_at",""))[:16].replace("T"," ")
        flags    = r.get("flags") or []
        if isinstance(flags, str):
            try: flags = json.loads(flags)
            except: flags = []
        color = RISK_COLOR.get(level, "#94a3b8")
        rid   = r.get("id")

        table_rows += f"""<tr>
          <td style="font-size:12px;color:#64748b">{dt}</td>
          <td title="{fname}">{fname[:28]}{"…" if len(fname)>28 else ""}</td>
          <td><span class="tag">{dtype}</span></td>
          <td>{applicant_id}</td>
          <td style="font-family:monospace;font-size:11px">{id_num[:14] if id_num!="—" else "—"}</td>
          <td style="font-weight:700;color:{color}">{level}</td>
          <td style="text-align:center">{score}</td>
          <td style="text-align:center">{_badge(level)}</td>
          <td style="text-align:center">{len(flags)}</td>
          <td>
            <a href="/portal/report/{rid}" class="btn btn-gray"
               style="padding:3px 10px;font-size:11px">Report</a>
          </td>
        </tr>"""

    content = f"""
    {alert}
    <div class="card">
      <form method="get" style="display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap">
        <div style="flex:1;min-width:160px">
          <label>Applicant ID</label>
          <input name="applicant" value="{applicant}" placeholder="Search applicant...">
        </div>
        <div style="min-width:130px">
          <label>Risk Level</label>
          <select name="risk">
            <option value="">All</option>
            <option {"selected" if risk=="HIGH" else ""}>HIGH</option>
            <option {"selected" if risk=="MEDIUM" else ""}>MEDIUM</option>
            <option {"selected" if risk=="LOW" else ""}>LOW</option>
          </select>
        </div>
        <div style="min-width:130px">
          <label>Doc Type</label>
          <select name="doc_type">
            <option value="">All</option>
            <option value="id_card" {"selected" if doc_type=="id_card" else ""}>ID Card</option>
            <option value="pdf" {"selected" if doc_type=="pdf" else ""}>PDF</option>
            <option value="image" {"selected" if doc_type=="image" else ""}>Image</option>
          </select>
        </div>
        <div style="min-width:100px">
          <label>Last N days</label>
          <select name="days">
            <option {"selected" if days==7 else ""} value="7">7 days</option>
            <option {"selected" if days==30 else ""} value="30">30 days</option>
            <option {"selected" if days==90 else ""} value="90">90 days</option>
          </select>
        </div>
        <button type="submit" class="btn">Filter</button>
        <a href="/portal/cases" class="btn btn-gray">Reset</a>
      </form>
    </div>

    <div class="card">
      <table>
        <thead><tr>
          <th>Date</th><th>File</th><th>Type</th><th>Applicant</th>
          <th>ID Number</th><th>Risk</th><th>Score</th><th>Action</th><th>Flags</th><th></th>
        </tr></thead>
        <tbody>
          {table_rows if table_rows else
           '<tr><td colspan="10" style="text-align:center;padding:30px;color:#94a3b8">No records found</td></tr>'}
        </tbody>
      </table>
    </div>"""

    return HTMLResponse(_shell("Cases", content, "cases"))


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLIANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/report/{record_id}", response_class=HTMLResponse)
def view_report(record_id: int):
    rows = _q("SELECT * FROM screening_logs WHERE id = :id", {"id": record_id})
    if not rows:
        raise HTTPException(404, "Record not found")
    r      = rows[0]
    result = r.get("full_result") or {}
    if isinstance(result, str):
        try: result = json.loads(result)
        except: result = {}

    level   = r.get("risk_level", "—")
    score   = r.get("risk_score", 0)
    flags   = r.get("flags") or []
    if isinstance(flags, str):
        try: flags = json.loads(flags)
        except: flags = []

    color   = RISK_COLOR.get(level, "#94a3b8")
    action  = {"HIGH":"REJECT","MEDIUM":"REVIEW","LOW":"PASS"}.get(level, "—")
    screened = str(r.get("screened_at",""))[:19].replace("T"," ")

    flag_html = "".join(f'<span class="tag">{f}</span>' for f in flags) or "<span style='color:#94a3b8'>None</span>"

    # field info
    fi     = result.get("field_info", {})
    ela    = result.get("ela", {})
    holo   = result.get("hologram", {})
    tmpl   = result.get("template_match", {})
    ml     = result.get("ml_inference", {})
    face   = result.get("face_match", {})

    def _row(label, value, highlight=False):
        style = "color:#ef4444;font-weight:600" if highlight else ""
        return f"<tr><td style='color:#64748b;width:40%'>{label}</td><td style='{style}'>{value}</td></tr>"

    content = f"""
    <div style="display:flex;gap:12px;margin-bottom:4px">
      <a href="/portal/cases" class="btn btn-gray" style="padding:6px 14px;font-size:12px">← Back to Cases</a>
      <a href="/portal/report/{record_id}/pdf" class="btn btn-green"
         style="padding:6px 14px;font-size:12px">⬇ Download PDF Report</a>
    </div>

    <div class="card" style="border-top:4px solid {color}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div style="font-size:22px;font-weight:700;color:{color}">{level} RISK</div>
          <div style="color:#64748b;margin-top:4px;font-size:13px">
            Score: <strong>{score}</strong> · Action: <strong>{action}</strong> · {screened}
          </div>
        </div>
        <div style="font-size:40px">{_badge(level)}</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3 style="font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
                   letter-spacing:.4px;margin-bottom:12px">Document Information</h3>
        <table>
          {_row("File Name", r.get("file_name","—"))}
          {_row("Document Type", r.get("doc_type","—").upper())}
          {_row("Applicant ID", r.get("applicant_id") or "—")}
          {_row("ID Number", fi.get("id_number") or "—")}
          {_row("Date of Birth", fi.get("dob") or "—")}
          {_row("Expiry Date", fi.get("expiry_date") or "—",
                highlight=fi.get("expiry_date") is not None and "expired" in str(flags))}
          {_row("MRZ Checksum", "✅ Valid" if fi.get("mrz_checksum_ok") else "❌ Failed" if fi.get("mrz_checksum_ok") is False else "—")}
          {_row("Screened At", screened)}
          {_row("SHA256", (r.get("file_sha256") or "—")[:20]+"…")}
        </table>
      </div>

      <div class="card">
        <h3 style="font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
                   letter-spacing:.4px;margin-bottom:12px">Forensic Scores</h3>
        <table>
          {_row("ELA Mean Diff", ela.get("ela_mean_diff","—"))}
          {_row("ELA Std Diff", ela.get("ela_std_diff","—"))}
          {_row("Hologram Patch", "✅ Detected" if holo.get("holographic_patch_detected") else "❌ Not detected")}
          {_row("FFT Peak Ratio", holo.get("fft_peak_ratio","—"))}
          {_row("Template Match", tmpl.get("template_matched","—"))}
          {_row("Keyword Ratio", tmpl.get("keyword_ratio","—"))}
          {_row("ML Prediction", ml.get("ml_prediction","—"))}
          {_row("ML Tamper Score", ml.get("ml_tamper_score","—"))}
          {_row("Face Match", "✅ Verified" if face.get("face_match") else "❌ Failed" if face.get("face_match") is False else "—")}
          {_row("Face Similarity", str(face.get("similarity_pct","—"))+"%"
                if face.get("similarity_pct") else "—")}
        </table>
      </div>
    </div>

    <div class="card">
      <h3 style="font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
                 letter-spacing:.4px;margin-bottom:12px">
        Fraud Flags ({len(flags)})
      </h3>
      {flag_html}
    </div>

    <div class="card">
      <h3 style="font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
                 letter-spacing:.4px;margin-bottom:12px">Staff Decision</h3>
      <form method="post" action="/portal/report/{record_id}/decision">
        <div class="grid-2">
          <div class="form-group">
            <label>Decision</label>
            <select name="decision">
              <option>PENDING REVIEW</option>
              <option>APPROVED — OVERRIDE</option>
              <option>REJECTED — CONFIRMED</option>
              <option>ESCALATED</option>
            </select>
          </div>
          <div class="form-group">
            <label>Staff Notes</label>
            <input type="text" name="notes" placeholder="Add notes for audit trail...">
          </div>
        </div>
        <button type="submit" class="btn">Save Decision</button>
      </form>
    </div>"""

    return HTMLResponse(_shell(f"Report #{record_id}", content, "cases"))


@app.post("/report/{record_id}/decision")
async def save_decision(record_id: int, decision: str = Form(""), notes: str = Form("")):
    db = SessionLocal()
    try:
        db.execute(text("""
            UPDATE screening_logs
            SET full_result = jsonb_set(
                COALESCE(full_result, '{{}}')::jsonb,
                '{{staff_decision}}',
                :val::jsonb
            )
            WHERE id = :id
        """), {
            "id": record_id,
            "val": json.dumps({
                "decision": decision,
                "notes": notes,
                "decided_at": datetime.utcnow().isoformat(),
            })
        })
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"Decision save failed: {e}")
    finally:
        db.close()
    return RedirectResponse(f"/portal/report/{record_id}?msg=Decision+saved", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/batch", response_class=HTMLResponse)
def batch_page(msg: str = ""):
    alert = f'<div class="alert alert-green">✅ {msg}</div>' if msg else ""

    content = f"""
    {alert}
    <div class="grid-2">
      <div class="card">
        <h3 style="font-size:14px;font-weight:600;margin-bottom:16px">📦 Batch ID Card Screening</h3>
        <p style="font-size:13px;color:#64748b;margin-bottom:14px">
          Upload a ZIP file containing ID card images. Each image will be
          screened automatically. Name files as <code>APPLICANT_ID.jpg</code>
          (e.g. <code>APP001.jpg</code>) to auto-populate applicant IDs.
        </p>
        <form action="/portal/batch/submit" method="post" enctype="multipart/form-data">
          <div class="form-group">
            <label>ZIP File containing ID card images *</label>
            <input type="file" name="zipfile" accept=".zip" required>
          </div>
          <div class="form-group">
            <label>Callback URL (optional — receive results via webhook)</label>
            <input type="text" name="callback_url"
                   placeholder="http://your-system/webhook"
                   value="http://172.16.26.48:3000/webhook/kyc">
          </div>
          <button type="submit" class="btn">Start Batch</button>
        </form>
      </div>

      <div class="card">
        <h3 style="font-size:14px;font-weight:600;margin-bottom:16px">📋 CSV Batch Submission</h3>
        <p style="font-size:13px;color:#64748b;margin-bottom:14px">
          Upload a CSV with applicant data. Each row triggers a screening job.
          CSV format: <code>applicant_id, file_path, doc_type</code>
        </p>
        <div class="alert alert-yellow" style="margin-bottom:14px">
          ⚠ Files must be accessible on the server at the paths specified in the CSV.
        </div>
        <form action="/portal/batch/csv" method="post" enctype="multipart/form-data">
          <div class="form-group">
            <label>CSV File *</label>
            <input type="file" name="csvfile" accept=".csv" required>
          </div>
          <button type="submit" class="btn">Process CSV</button>
        </form>
        <div style="margin-top:12px">
          <a href="/portal/batch/sample.csv" class="btn btn-gray"
             style="padding:5px 12px;font-size:11px">⬇ Download Sample CSV</a>
        </div>
      </div>
    </div>

    <div class="card">
      <h3 style="font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
                 letter-spacing:.4px;margin-bottom:12px">Recent Batch Jobs</h3>
      <table>
        <thead><tr><th>Date</th><th>Batch ID</th><th>Total</th><th>High</th><th>Medium</th><th>Low</th><th>Status</th></tr></thead>
        <tbody>
          <tr><td colspan="7" style="text-align:center;padding:20px;color:#94a3b8">
            Batch history coming soon
          </td></tr>
        </tbody>
      </table>
    </div>"""

    return HTMLResponse(_shell("Batch Processing", content, "batch"))


@app.get("/batch/sample.csv")
def sample_csv():
    from fastapi.responses import Response
    content = "applicant_id,file_path,doc_type\nAPP001,/app/test_files/id1.jpg,id_card\nAPP002,/app/test_files/statement.pdf,pdf\n"
    return Response(content, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_batch.csv"})


@app.post("/batch/submit")
async def batch_submit(
    zipfile: UploadFile = File(...),
    callback_url: str = Form(""),
):
    import zipfile as zf
    import tempfile
    import httpx

    submitted = 0
    errors    = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "batch.zip"
        zip_path.write_bytes(await zipfile.read())

        try:
            with zf.ZipFile(str(zip_path)) as z:
                names = [n for n in z.namelist()
                         if n.lower().endswith((".jpg",".jpeg",".png"))]

                async with httpx.AsyncClient(base_url="http://api:8001") as client:
                    for name in names:
                        try:
                            img_data    = z.read(name)
                            applicant_id = Path(name).stem
                            data = {"applicant_id": applicant_id}
                            if callback_url:
                                data["callback_url"] = callback_url
                            resp = await client.post("/screen-id-card",
                                files={"file": (name, img_data, "image/jpeg")},
                                data=data, timeout=30)
                            if resp.status_code == 202:
                                submitted += 1
                            else:
                                errors += 1
                        except Exception as e:
                            log.error(f"Batch item {name} failed: {e}")
                            errors += 1
        except zf.BadZipFile:
            return RedirectResponse("/portal/batch?msg=Invalid+ZIP+file", 303)

    return RedirectResponse(
        f"/portal/batch?msg=Batch+complete:+{submitted}+submitted,+{errors}+errors", 303)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPIRY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/expiry", response_class=HTMLResponse)
def expiry_page():
    # Find IDs with expiry dates from the database
    rows = _q("""
        SELECT id, file_name, applicant_id, id_number,
               full_result->>'screened_at' AS screened_at,
               full_result->'field_info'->>'expiry_date' AS expiry_date,
               risk_level
        FROM screening_logs
        WHERE doc_type = 'id_card'
        AND full_result->'field_info'->>'expiry_date' IS NOT NULL
        ORDER BY screened_at DESC
        LIMIT 100
    """)

    now     = datetime.utcnow()
    expired, expiring30, expiring90, valid = [], [], [], []

    for r in rows:
        exp_str = r.get("expiry_date", "")
        if not exp_str:
            continue
        try:
            from dateutil import parser as dp
            exp_dt = dp.parse(exp_str, dayfirst=True)
            days_left = (exp_dt - now).days
            r["days_left"]  = days_left
            r["expiry_fmt"] = exp_dt.strftime("%d %b %Y")
            if days_left < 0:
                expired.append(r)
            elif days_left <= 30:
                expiring30.append(r)
            elif days_left <= 90:
                expiring90.append(r)
            else:
                valid.append(r)
        except Exception:
            continue

    def _exp_table(items, color, label):
        if not items:
            return f'<p style="color:#94a3b8;font-size:13px">No {label} IDs</p>'
        trs = ""
        for r in items:
            days = r.get("days_left", 0)
            days_str = f'<span style="color:{color};font-weight:700">' + \
                       (f"Expired {abs(days)}d ago" if days < 0 else f"{days}d left") + \
                       "</span>"
            trs += f"""<tr>
              <td>{r.get("applicant_id") or "—"}</td>
              <td>{r.get("id_number") or "—"}</td>
              <td>{r.get("expiry_fmt","—")}</td>
              <td>{days_str}</td>
              <td>{r.get("risk_level","—")}</td>
              <td><a href="/portal/report/{r.get('id')}" class="btn btn-gray"
                     style="padding:3px 10px;font-size:11px">View</a></td>
            </tr>"""
        return f"""<table>
          <thead><tr><th>Applicant</th><th>ID Number</th><th>Expiry</th>
                     <th>Status</th><th>Risk</th><th></th></tr></thead>
          <tbody>{trs}</tbody></table>"""

    content = f"""
    <div class="grid-3" style="margin-bottom:20px">
      <div class="stat">
        <div class="num" style="color:#ef4444">{len(expired)}</div>
        <div class="lbl">Expired IDs</div>
      </div>
      <div class="stat">
        <div class="num" style="color:#f59e0b">{len(expiring30)}</div>
        <div class="lbl">Expiring in 30 days</div>
      </div>
      <div class="stat">
        <div class="num" style="color:#3b82f6">{len(expiring90)}</div>
        <div class="lbl">Expiring in 90 days</div>
      </div>
    </div>

    <div class="card" style="border-left:4px solid #ef4444">
      <h3 style="font-size:13px;font-weight:600;color:#ef4444;margin-bottom:12px">
        🔴 Expired IDs ({len(expired)})
      </h3>
      {_exp_table(expired, "#ef4444", "expired")}
    </div>

    <div class="card" style="border-left:4px solid #f59e0b">
      <h3 style="font-size:13px;font-weight:600;color:#f59e0b;margin-bottom:12px">
        🟡 Expiring Within 30 Days ({len(expiring30)})
      </h3>
      {_exp_table(expiring30, "#f59e0b", "expiring soon")}
    </div>

    <div class="card" style="border-left:4px solid #3b82f6">
      <h3 style="font-size:13px;font-weight:600;color:#3b82f6;margin-bottom:12px">
        🔵 Expiring Within 90 Days ({len(expiring90)})
      </h3>
      {_exp_table(expiring90, "#3b82f6", "expiring in 90 days")}
    </div>"""

    return HTMLResponse(_shell("Expiry Tracker", content, "expiry"))


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "running", "service": "portal"}
