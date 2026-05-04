"""
cbc_portal.py — CBC Credit Analysis Portal
Web interface for uploading CBC PDFs, extracting data,
previewing results, and downloading the filled Excel template.

Runs on port 8007. Access at /cbc/ via Nginx.
Integrated into the existing loan officer portal system.
"""

import os
import io
import json
import uuid
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("cbc_portal")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads")) / "cbc"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CBC Credit Analysis Portal", version="1.0.0")

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
:root{
  --nav:#1a2744;--accent:#2563eb;--accent-l:#eff6ff;
  --ok:#059669;--ok-bg:#ecfdf5;--warn:#d97706;--warn-bg:#fffbeb;
  --danger:#dc2626;--danger-bg:#fef2f2;
  --surface:#f8fafc;--card:#fff;--border:#e2e8f0;
  --text:#0f172a;--muted:#64748b;
  --green-dark:#1F5C2E;--green-light:#C6EFCE;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);font-size:14px}
.topbar{background:var(--nav);color:#fff;height:56px;display:flex;
        align-items:center;padding:0 24px;justify-content:space-between;
        box-shadow:0 2px 8px rgba(0,0,0,.2);position:sticky;top:0;z-index:100}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;background:var(--green-dark);border-radius:7px;
            display:flex;align-items:center;justify-content:center;
            font-size:13px;font-weight:800;color:#fff}
.brand-title{font-size:14px;font-weight:700;color:#f1f5f9}
.brand-sub{font-size:11px;color:#64748b}
.nav-links{display:flex;gap:8px}
.nav-link{color:#94a3b8;font-size:12px;padding:6px 12px;border-radius:6px;
          text-decoration:none;transition:all .15s}
.nav-link:hover{background:rgba(255,255,255,.1);color:#fff}
.content{max-width:1200px;margin:0 auto;padding:24px 20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:22px 24px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.card-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;
            letter-spacing:.6px;margin-bottom:16px;padding-bottom:10px;
            border-bottom:1px solid var(--border)}
.upload-zone{border:2px dashed #cbd5e1;border-radius:12px;padding:40px 20px;
             text-align:center;cursor:pointer;transition:all .2s;background:#f8fafc}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:var(--accent-l)}
.upload-icon{font-size:36px;margin-bottom:12px}
.upload-title{font-size:16px;font-weight:600;color:var(--text);margin-bottom:6px}
.upload-sub{font-size:13px;color:var(--muted)}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 20px;
     border-radius:8px;font-size:13px;font-weight:500;cursor:pointer;
     border:none;transition:all .15s;text-decoration:none}
.btn-primary{background:var(--green-dark);color:#fff}
.btn-primary:hover{background:#164022}
.btn-accent{background:var(--accent);color:#fff}
.btn-accent:hover{background:#1d4ed8}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface)}
.btn-success{background:var(--ok);color:#fff}
.btn-lg{padding:13px 28px;font-size:14px;font-weight:600}
.progress{height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;margin:12px 0}
.progress-bar{height:100%;background:var(--accent);border-radius:3px;
              transition:width .4s ease;width:0%}
.alert{padding:12px 16px;border-radius:8px;font-size:13px;margin-bottom:14px;
       display:flex;align-items:flex-start;gap:10px}
.alert-ok{background:var(--ok-bg);border:1px solid #6ee7b7;color:#065f46}
.alert-err{background:var(--danger-bg);border:1px solid #fca5a5;color:#991b1b}
.alert-info{background:var(--accent-l);border:1px solid #93c5fd;color:#1e40af}

/* ── CBC Preview Table ── */
.cbc-header{background:var(--green-dark);color:#fff;padding:10px 14px;
            border-radius:8px 8px 0 0;font-weight:600;font-size:13px}
.cbc-info{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;
          background:#f0fdf4;border:1px solid #bbf7d0;padding:12px 14px}
.cbc-info-item{font-size:12px}
.cbc-info-label{color:var(--muted);font-weight:500;margin-bottom:2px}
.cbc-info-val{font-weight:600;color:var(--text)}
.summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.sum-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
          padding:14px 16px;text-align:center}
.sum-num{font-size:24px;font-weight:700;margin-bottom:3px}
.sum-lbl{font-size:11px;color:var(--muted);font-weight:500}
.section-hdr{background:var(--green-dark);color:#fff;font-weight:600;font-size:12px;
             padding:8px 12px;margin:16px 0 0;border-radius:6px 6px 0 0}
.accounts-table{width:100%;border-collapse:collapse;font-size:11px;margin-bottom:16px}
.accounts-table th{background:#dbeafe;color:#1e40af;padding:7px 8px;text-align:left;
                   font-size:10px;font-weight:700;text-transform:uppercase;
                   letter-spacing:.3px;border-bottom:2px solid #93c5fd;
                   white-space:nowrap}
.accounts-table td{padding:7px 8px;border-bottom:1px solid #f1f5f9;vertical-align:middle}
.accounts-table tr:hover td{background:#f8fafc}
.accounts-table tr.closed td{color:var(--muted)}
.pill{display:inline-flex;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
.pill-normal{background:#dcfce7;color:#166534}
.pill-closed{background:#f1f5f9;color:#64748b}
.pill-writeoff{background:#fef2f2;color:#dc2626}
.pill-delinquent{background:#fffbeb;color:#92400e}
.cycle-bar{font-family:monospace;font-size:10px;letter-spacing:1px;color:#374151}
.tab-row{display:flex;gap:4px;margin-bottom:0;border-bottom:2px solid var(--border)}
.tab{padding:10px 16px;font-size:13px;font-weight:500;cursor:pointer;
     border-radius:8px 8px 0 0;background:transparent;border:none;color:var(--muted);
     transition:all .15s}
.tab.active{background:var(--green-dark);color:#fff}
.tab-content{display:none}.tab-content.active{display:block}
.step-indicator{display:flex;align-items:center;gap:0;margin-bottom:24px}
.step{display:flex;align-items:center;flex:1}
.step-dot{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;
          justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;
          border:2px solid var(--border);background:var(--card);color:var(--muted)}
.step.done .step-dot{background:var(--ok);border-color:var(--ok);color:#fff}
.step.active .step-dot{background:var(--accent);border-color:var(--accent);color:#fff}
.step-label{font-size:11px;color:var(--muted);margin-left:8px;font-weight:500}
.step.active .step-label{color:var(--accent);font-weight:700}
.step.done .step-label{color:var(--ok)}
.step-line{flex:1;height:2px;background:var(--border);margin:0 8px}
.step.done .step-line{background:var(--ok)}
.amount{font-family:monospace;font-size:11px}
"""

HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>CBC Credit Analysis — {title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="brand-icon">CBC</div>
      <div>
        <div class="brand-title">CBC Credit Analysis</div>
        <div class="brand-sub">Cambodia Credit Bureau — PDF to Excel</div>
      </div>
    </div>
    <div class="nav-links">
      <a href="/cbc/" class="nav-link">Upload</a>
      <a href="/loan/" class="nav-link">Loan Portal</a>
      <a href="/portal/" class="nav-link">KYC Portal</a>
    </div>
  </div>
  <div class="content">
    {body}
  </div>
  {scripts}
</body>
</html>"""


def _shell(title, body, scripts=""):
    return HTML_SHELL.format(title=title, css=CSS, body=body, scripts=scripts)


def _pill(status: str) -> str:
    s = (status or "").lower()
    cls = {"normal":"pill-normal","closed":"pill-closed",
           "write off":"pill-writeoff","delinquent":"pill-delinquent"}.get(s,"pill-closed")
    return f'<span class="pill {cls}">{status}</span>'


def _fmt_amount(amount, currency="USD") -> str:
    try:
        v = float(amount)
        if v == 0: return "—"
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
        return f"{currency} {s}"
    except:
        return str(amount)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    body = """
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
      <div>
        <h1 style="font-size:20px;font-weight:700">CBC Report Extractor</h1>
        <p style="font-size:13px;color:var(--muted);margin-top:3px">
          Upload a CBC PDF report → auto-extract all loan accounts → download filled Excel template
        </p>
      </div>
    </div>

    <div class="step-indicator">
      <div class="step active">
        <div class="step-dot">1</div>
        <div class="step-label">Upload PDF</div>
        <div class="step-line"></div>
      </div>
      <div class="step">
        <div class="step-dot">2</div>
        <div class="step-label">Extract & Review</div>
        <div class="step-line"></div>
      </div>
      <div class="step">
        <div class="step-dot">3</div>
        <div class="step-label">Download Excel</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Upload CBC Report PDF</div>

      <div class="alert alert-info">
        <span>ℹ</span>
        <div>
          Upload the CBC PDF report downloaded from
          <strong>pr.creditbureau.com.kh</strong>.
          The system will extract all applicant data, loan accounts,
          and payment history — then generate the bank's uniform Excel summary.
        </div>
      </div>

      <form id="upload-form" action="/cbc/extract" method="post"
            enctype="multipart/form-data">
        <div class="upload-zone" id="drop-zone"
             onclick="document.getElementById('pdf-input').click()">
          <div class="upload-icon">📄</div>
          <div class="upload-title">Drop CBC PDF here or click to browse</div>
          <div class="upload-sub">Supports CBC Consumer Credit Report PDFs · Max 20MB</div>
          <input type="file" id="pdf-input" name="pdf_file"
                 accept=".pdf" style="display:none"
                 onchange="handleFileSelect(this)">
        </div>

        <div id="file-info" style="display:none;margin-top:12px">
          <div class="alert alert-ok" id="file-name-display"></div>
          <div class="progress"><div class="progress-bar" id="prog-bar"></div></div>
        </div>

        <div style="margin-top:16px;display:flex;gap:10px">
          <button type="submit" class="btn btn-primary btn-lg" id="submit-btn"
                  disabled onclick="startUpload()">
            Extract & Generate Excel
          </button>
          <button type="button" class="btn btn-ghost" onclick="resetForm()">
            Clear
          </button>
        </div>
      </form>
    </div>

    <div class="card">
      <div class="card-title">What gets extracted</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
        <div style="padding:14px;background:#f0fdf4;border-radius:8px;border:1px solid #bbf7d0">
          <div style="font-size:20px;margin-bottom:8px">👤</div>
          <div style="font-weight:600;margin-bottom:4px;font-size:13px">Applicant Details</div>
          <div style="font-size:12px;color:var(--muted)">
            Name, ID number, DOB, gender, nationality, address, employment, income
          </div>
        </div>
        <div style="padding:14px;background:#eff6ff;border-radius:8px;border:1px solid #bfdbfe">
          <div style="font-size:20px;margin-bottom:8px">🏦</div>
          <div style="font-weight:600;margin-bottom:4px;font-size:13px">Loan Accounts</div>
          <div style="font-size:12px;color:var(--muted)">
            All active, closed, and write-off accounts with creditor, limit,
            outstanding balance, installment, tenure, payment history
          </div>
        </div>
        <div style="padding:14px;background:#fef3c7;border-radius:8px;border:1px solid #fcd34d">
          <div style="font-size:20px;margin-bottom:8px">📊</div>
          <div style="font-weight:600;margin-bottom:4px;font-size:13px">Credit Summary</div>
          <div style="font-size:12px;color:var(--muted)">
            Total accounts, normal/delinquent/closed counts, total limits,
            total liabilities, payment cycle history, guaranteed accounts
          </div>
        </div>
      </div>
    </div>"""

    scripts = """<script>
const dropZone = document.getElementById('drop-zone');
const input    = document.getElementById('pdf-input');

dropZone.addEventListener('dragover', e => {
  e.preventDefault(); dropZone.classList.add('drag');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  if (e.dataTransfer.files.length) {
    input.files = e.dataTransfer.files;
    handleFileSelect(input);
  }
});

function handleFileSelect(inp) {
  const f = inp.files[0];
  if (!f) return;
  document.getElementById('file-info').style.display = 'block';
  document.getElementById('file-name-display').innerHTML =
    '✓ Selected: <strong>' + f.name + '</strong> (' +
    (f.size/1024/1024).toFixed(2) + ' MB)';
  document.getElementById('submit-btn').disabled = false;
}

function resetForm() {
  document.getElementById('upload-form').reset();
  document.getElementById('file-info').style.display = 'none';
  document.getElementById('submit-btn').disabled = true;
}

function startUpload() {
  const btn = document.getElementById('submit-btn');
  btn.textContent = 'Extracting...';
  btn.disabled = true;
  let pct = 0;
  const bar = document.getElementById('prog-bar');
  const iv = setInterval(() => {
    pct = Math.min(pct + Math.random()*15, 85);
    bar.style.width = pct + '%';
  }, 300);
}
</script>"""

    return HTMLResponse(_shell("Upload", body, scripts))


@app.post("/extract", response_class=HTMLResponse)
async def extract(pdf_file: UploadFile = File(...)):
    """Upload PDF and redirect to progress page."""
    uid      = str(uuid.uuid4())[:8]
    pdf_path = UPLOAD_DIR / f"{uid}_{pdf_file.filename}"
    content  = await pdf_file.read()
    pdf_path.write_bytes(content)

    job_file = UPLOAD_DIR / f"{uid}_job.json"
    job_file.write_text(json.dumps({
        "uid":      uid,
        "filename": pdf_file.filename,
        "pdf_path": str(pdf_path),
        "size_mb":  round(len(content)/1024/1024, 2),
    }))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(f"/cbc/progress/{uid}", status_code=303)


@app.get("/progress/{uid}", response_class=HTMLResponse)
def progress_page(uid: str):
    body = f"""
    <div style="max-width:700px;margin:0 auto">
      <h1 style="font-size:20px;font-weight:700;margin-bottom:6px">Processing CBC Report</h1>
      <p style="font-size:13px;color:var(--muted);margin-bottom:20px">
        Extracting credit data from CBC PDF...
      </p>
      <div class="card">
        <div class="card-title">Extraction Progress</div>
        <div class="progress" style="height:10px;margin-bottom:16px">
          <div class="progress-bar" id="pbar" style="transition:width .5s"></div>
        </div>
        <div id="status-text" style="font-size:13px;font-weight:600;color:var(--accent);margin-bottom:12px">
          Starting...
        </div>
        <div id="log-box" style="
          background:#0f172a;color:#e2e8f0;font-family:monospace;font-size:12px;
          padding:16px;border-radius:8px;height:280px;overflow-y:auto;
          line-height:1.6;white-space:pre-wrap"></div>
      </div>
    </div>
    <script>
    const logBox  = document.getElementById('log-box');
    const pbar    = document.getElementById('pbar');
    const statusT = document.getElementById('status-text');
    function appendLog(msg, color) {{
      const s = document.createElement('span');
      s.style.color = color || '#e2e8f0';
      s.textContent = msg + '\\n';
      logBox.appendChild(s);
      logBox.scrollTop = logBox.scrollHeight;
    }}
    function setProgress(pct, label) {{
      pbar.style.width = pct + '%';
      if (label) statusT.textContent = label;
    }}
    const es = new EventSource('/cbc/stream/{uid}');
    es.addEventListener('log',      e => {{ const d=JSON.parse(e.data); appendLog(d.msg, d.color); }});
    es.addEventListener('progress', e => {{ const d=JSON.parse(e.data); setProgress(d.pct, d.label); }});
    es.addEventListener('done',     e => {{
      es.close(); setProgress(100, '✅ Complete — redirecting...');
      appendLog('\\n✅ Done!', '#4ade80');
      setTimeout(() => window.location.href = '/cbc/result/{uid}', 800);
    }});
    es.addEventListener('error_msg', e => {{
      es.close(); const d=JSON.parse(e.data);
      setProgress(0,'❌ Failed'); appendLog('\\n❌ ' + d.msg, '#f87171');
    }});
    </script>"""
    return HTMLResponse(_shell("Processing", body))


@app.get("/stream/{uid}")
async def stream(uid: str):
    import asyncio
    from fastapi.responses import StreamingResponse as SR

    job_file = UPLOAD_DIR / f"{uid}_job.json"
    if not job_file.exists():
        async def err():
            yield f"event: error_msg\ndata: {{\"msg\": \"Job not found\"}}\n\n"
        return SR(err(), media_type="text/event-stream")

    job      = json.loads(job_file.read_text())
    pdf_path = job["pdf_path"]

    async def generate():
        def evt(event, data): return f"event: {event}\ndata: {json.dumps(data)}\n\n"
        def log(msg, color="#e2e8f0"): return evt("log", {"msg": msg, "color": color})
        def prog(pct, label):          return evt("progress", {"pct": pct, "label": label})

        try:
            yield log(f"📄 File: {job['filename']} ({job['size_mb']} MB)", "#94a3b8")
            yield prog(5, "Opening PDF...")

            import concurrent.futures
            loop = asyncio.get_event_loop()
            q = []

            def run():
                from cbc_extractor import extract_cbc_data, fill_excel_template
                q.append(("log","🔍 Parsing CBC report...","#93c5fd"))
                q.append(("prog", 15, "Reading applicant data..."))

                data = extract_cbc_data(pdf_path)
                h    = data["header"]
                apps = data["applicants"]

                q.append(("log", f"   Enquiry No:  {h.get('enquiry_number','')}", "#e2e8f0"))
                q.append(("log", f"   Report Date: {h.get('report_date','')}", "#e2e8f0"))
                q.append(("log", f"   Applicants:  {len(apps)}", "#e2e8f0"))
                q.append(("prog", 40, "Extracting account details..."))

                for app in apps:
                    p  = app["personal"]
                    s  = app["summary"]
                    ac = app["accounts"]
                    q.append(("log", f"\n👤 Applicant {app['index']}: {p.get('full_name_en','?')}", "#4ade80"))
                    q.append(("log", f"   Total accounts: {s.get('total_accounts',0)}", "#e2e8f0"))
                    q.append(("log", f"   Normal/Closed:  {s.get('normal_accounts',0)} / {s.get('closed_accounts',0)}", "#e2e8f0"))
                    q.append(("log", f"   Active found:   {len(ac.get('active',[]))}", "#e2e8f0"))
                    q.append(("log", f"   Closed found:   {len(ac.get('closed',[]))}", "#e2e8f0"))

                q.append(("prog", 75, "Generating Excel template..."))
                xlsx_path = str(UPLOAD_DIR / f"{uid}_CBC_Summary.xlsx")
                fill_excel_template(data, xlsx_path)
                q.append(("log", f"\n📊 Excel generated successfully", "#4ade80"))
                q.append(("prog", 95, "Saving..."))

                # Save result for result page
                result_file = UPLOAD_DIR / f"{uid}_result.json"
                result_file.write_text(json.dumps({
                    "uid": uid, "header": h,
                    "applicants": [
                        {
                            "index":    a["index"],
                            "personal": a["personal"],
                            "summary":  a["summary"],
                            "accounts": {
                                k: [{kk: str(vv) if hasattr(vv,'strftime') else vv
                                     for kk,vv in acc.items()}
                                    for acc in v]
                                for k,v in a["accounts"].items()
                            }
                        }
                        for a in apps
                    ]
                }, default=str))
                q.append(("done", None, None))

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = loop.run_in_executor(pool, run)
                while not future.done():
                    while q:
                        item = q.pop(0)
                        if item[0] == "log":   yield log(item[1], item[2])
                        elif item[0] == "prog": yield prog(item[1], item[2])
                    await asyncio.sleep(0.05)
                while q:
                    item = q.pop(0)
                    if item[0] == "log":   yield log(item[1], item[2])
                    elif item[0] == "prog": yield prog(item[1], item[2])
                await future

            yield prog(100, "Complete!")
            yield evt("done", {})

        except Exception as e:
            import traceback
            yield log(f"\n❌ {e}", "#f87171")
            yield log(traceback.format_exc(), "#f87171")
            yield evt("error_msg", {"msg": str(e)})

    return SR(generate(), media_type="text/event-stream",
              headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.get("/result/{uid}", response_class=HTMLResponse)
def result_page(uid: str):
    result_file = UPLOAD_DIR / f"{uid}_result.json"
    if not result_file.exists():
        return HTMLResponse(_shell("Error",
            '<div class="alert alert-err">Result not found.</div>'))

    result     = json.loads(result_file.read_text())
    header     = result.get("header", {})
    applicants = result.get("applicants", [])

    tabs_html   = '<div class="tab-row">'
    panels_html = ""
    for i, app in enumerate(applicants):
        p    = app.get("personal", {})
        name = p.get("full_name_en","") or f"Applicant {app['index']}"
        active_cls = "active" if i == 0 else ""
        tabs_html   += f'<button class="tab {active_cls}" onclick="showTab({i})" id="tab-{i}">{name}</button>'
        panels_html += _build_applicant_panel(app, i, active_cls)
    tabs_html += "</div>"

    body = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
      <div>
        <h1 style="font-size:20px;font-weight:700">✅ CBC Extraction Complete</h1>
      </div>
      <div style="display:flex;gap:10px">
        <a href="/cbc/download/{uid}" class="btn btn-primary btn-lg">⬇ Download Excel</a>
        <a href="/cbc/" class="btn btn-ghost">Upload Another</a>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Report Header</div>
      <div class="cbc-info">
        <div class="cbc-info-item"><div class="cbc-info-label">Enquiry No.</div>
          <div class="cbc-info-val">{header.get('enquiry_number','—')}</div></div>
        <div class="cbc-info-item"><div class="cbc-info-label">Report Date</div>
          <div class="cbc-info-val">{header.get('report_date','—')}</div></div>
        <div class="cbc-info-item"><div class="cbc-info-label">Product Type</div>
          <div class="cbc-info-val">{header.get('product_type','—')}</div></div>
        <div class="cbc-info-item"><div class="cbc-info-label">Amount</div>
          <div class="cbc-info-val">{header.get('amount','—')}</div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Applicant Details</div>
      {tabs_html}
      <div style="padding-top:16px">{panels_html}</div>
    </div>

    <div style="text-align:center;padding:20px">
      <a href="/cbc/download/{uid}" class="btn btn-primary btn-lg">
        ⬇ Download Filled Excel Template
      </a>
    </div>"""

    scripts = """<script>
function showTab(idx) {
  document.querySelectorAll('.tab').forEach((t,i) => t.className='tab'+(i===idx?' active':''));
  document.querySelectorAll('.tab-content').forEach((p,i) => p.className='tab-content'+(i===idx?' active':''));
}
</script>"""
    return HTMLResponse(_shell("Results", body, scripts))

    # ── Build preview ─────────────────────────────────────────────────────────
    tabs_html  = '<div class="tab-row">'
    panels_html= ""
    for i, app in enumerate(applicants):
        p    = app["personal"]
        name = p.get("full_name_en","") or f"Applicant {app['index']}"
        active_cls = "active" if i == 0 else ""
        tabs_html += f'<button class="tab {active_cls}" onclick="showTab({i})" id="tab-{i}">{name}</button>'
        panels_html += _build_applicant_panel(app, i, active_cls)
    tabs_html += "</div>"

    body = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
      <div>
        <h1 style="font-size:20px;font-weight:700">Extraction Complete</h1>
        <p style="font-size:13px;color:var(--muted);margin-top:3px">
          Review extracted data then download the filled Excel template
        </p>
      </div>
      <div style="display:flex;gap:10px">
        <a href="/cbc/download/{uid}" class="btn btn-primary btn-lg">
          ⬇ Download Excel
        </a>
        <a href="/cbc/" class="btn btn-ghost">Upload Another</a>
      </div>
    </div>

    <div class="step-indicator">
      <div class="step done"><div class="step-dot">✓</div>
        <div class="step-label">Upload PDF</div><div class="step-line"></div></div>
      <div class="step done"><div class="step-dot">✓</div>
        <div class="step-label">Extract & Review</div><div class="step-line"></div></div>
      <div class="step active"><div class="step-dot">3</div>
        <div class="step-label">Download Excel</div></div>
    </div>

    <div class="card">
      <div class="card-title">Report Header</div>
      <div class="cbc-info">
        <div class="cbc-info-item">
          <div class="cbc-info-label">Enquiry Number</div>
          <div class="cbc-info-val">{header.get("enquiry_number","—")}</div>
        </div>
        <div class="cbc-info-item">
          <div class="cbc-info-label">Report Date</div>
          <div class="cbc-info-val">{header.get("report_date","—")}</div>
        </div>
        <div class="cbc-info-item">
          <div class="cbc-info-label">Product Type</div>
          <div class="cbc-info-val">{header.get("product_type","—")}</div>
        </div>
        <div class="cbc-info-item">
          <div class="cbc-info-label">Amount / Account Type</div>
          <div class="cbc-info-val">{header.get("amount","—")} · {header.get("account_type","—")}</div>
        </div>
        <div class="cbc-info-item">
          <div class="cbc-info-label">Member Reference</div>
          <div class="cbc-info-val">{header.get("member_ref","—")}</div>
        </div>
        <div class="cbc-info-item">
          <div class="cbc-info-label">Applicants</div>
          <div class="cbc-info-val">{len(applicants)}</div>
        </div>
        <div class="cbc-info-item">
          <div class="cbc-info-label">Enquiry Type</div>
          <div class="cbc-info-val">{header.get("enquiry_type","—")}</div>
        </div>
        <div class="cbc-info-item">
          <div class="cbc-info-label">Excel File</div>
          <div class="cbc-info-val" style="color:var(--ok)">✓ Ready to download</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Applicant Credit Summary</div>
      {tabs_html}
      <div style="padding-top:16px">
        {panels_html}
      </div>
    </div>

    <div style="text-align:center;padding:20px">
      <a href="/cbc/download/{uid}" class="btn btn-primary btn-lg">
        ⬇ Download Filled Excel Template
      </a>
      <p style="font-size:12px;color:var(--muted);margin-top:10px">
        Bank's uniform CBC Account Summary — ready for credit analysis
      </p>
    </div>"""

    scripts = """<script>
function showTab(idx) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.className = 'tab' + (i===idx?' active':'');
  });
  document.querySelectorAll('.tab-content').forEach((p,i) => {
    p.className = 'tab-content' + (i===idx?' active':'');
  });
}
</script>"""

    return HTMLResponse(_shell("Review", body, scripts))


def _build_applicant_panel(app: dict, idx: int, active_cls: str) -> str:
    p       = app["personal"]
    s       = app["summary"]
    accs    = app["accounts"]
    emp     = app.get("employment", [])

    name    = p.get("full_name_en","")
    active  = accs.get("active",[])
    closed  = accs.get("closed",[])
    guar_c  = accs.get("guaranteed_closed",[])

    def _accounts_table(rows: list, is_closed=False) -> str:
        if not rows: return "<p style='color:var(--muted);font-size:12px;padding:8px'>None</p>"
        html = """<table class="accounts-table">
        <thead><tr>
          <th>#</th><th>Creditor</th><th>Product</th><th>Status</th>
          <th>Currency</th><th>Limit</th><th>Outstanding</th><th>Installment</th>
          <th>Issue Date</th><th>Expiry Date</th>
          <th>Closed Date</th><th>Tenure(m)</th><th>Security</th>
          <th>Restructured</th><th>Last 24 Cycles</th><th>Advisory</th>
        </tr></thead><tbody>"""
        for i, a in enumerate(rows, 1):
            row_cls = "closed" if is_closed else ""
            st = "Closed" if is_closed else a.get("status","Normal")
            html += f"""<tr class="{row_cls}">
              <td>{i}</td>
              <td style="font-weight:500">{a.get("creditor","")}</td>
              <td>{a.get("product_type","")}</td>
              <td>{_pill(st)}</td>
              <td>{a.get("currency","")}</td>
              <td class="amount">{_fmt_amount(a.get("limit",0), a.get("currency","USD"))}</td>
              <td class="amount">{_fmt_amount(a.get("outstanding",0), a.get("currency","USD"))}</td>
              <td class="amount">{_fmt_amount(a.get("installment",0), a.get("currency","USD"))}</td>
              <td>{a.get("issue_date","")}</td>
              <td>{a.get("expiry_date","")}</td>
              <td>{a.get("closed_date","") or "—"}</td>
              <td style="text-align:center">{int(a.get("tenure",0)) or "—"}</td>
              <td style="font-size:10px">{a.get("security_type","")}</td>
              <td style="text-align:center">{a.get("restructured","No")}</td>
              <td class="cycle-bar">{a.get("last_24_cycles","")}</td>
              <td style="font-size:10px;color:var(--warn)">{a.get("advisory","")}</td>
            </tr>"""
        html += "</tbody></table>"
        return html

    # Employment info
    emp_html = ""
    if emp:
        latest = emp[0]
        income_raw = latest.get("income","0")
        cur_emp    = "KHR" if "riel" in latest.get("currency","").lower() else "USD"
        try:
            income_v = float(str(income_raw).replace(",",""))
            income_s = _fmt_amount(income_v, cur_emp)
        except:
            income_s = income_raw
        emp_html = f"""
        <div style="display:flex;gap:14px;font-size:12px;margin-bottom:12px;
             padding:10px 12px;background:#f8fafc;border-radius:6px;border:1px solid var(--border)">
          <div><span style="color:var(--muted)">Employer: </span>
               <strong>{latest.get("employer","—")}</strong></div>
          <div><span style="color:var(--muted)">Occupation: </span>
               {latest.get("occupation","—")}</div>
          <div><span style="color:var(--muted)">Type: </span>
               {latest.get("emp_type","—")}</div>
          <div><span style="color:var(--muted)">Monthly Income: </span>
               <strong style="color:var(--ok)">{income_s}</strong></div>
        </div>"""

    panel = f"""<div class="tab-content {active_cls}" id="panel-{idx}">

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px">
        <div style="padding:14px;background:#f8fafc;border-radius:8px;border:1px solid var(--border)">
          <div style="font-weight:600;margin-bottom:10px;font-size:13px">Personal Information</div>
          <table style="font-size:12px;width:100%">
            <tr><td style="color:var(--muted);padding:3px 0;width:40%">Full Name</td>
                <td><strong>{p.get("full_name_en","—")}</strong></td></tr>
            <tr><td style="color:var(--muted);padding:3px 0">ID Number</td>
                <td>{p.get("id_number","—")} ({p.get("id_type","").split()[0]})</td></tr>
            <tr><td style="color:var(--muted);padding:3px 0">Date of Birth</td>
                <td>{p.get("dob","—")}</td></tr>
            <tr><td style="color:var(--muted);padding:3px 0">Gender</td>
                <td>{p.get("gender","—")}</td></tr>
            <tr><td style="color:var(--muted);padding:3px 0">Marital Status</td>
                <td>{p.get("marital_status","—")}</td></tr>
            <tr><td style="color:var(--muted);padding:3px 0">Nationality</td>
                <td>{p.get("nationality","—")}</td></tr>
          </table>
        </div>
        <div style="padding:14px;background:#f8fafc;border-radius:8px;border:1px solid var(--border)">
          <div style="font-weight:600;margin-bottom:10px;font-size:13px">Credit Summary</div>
          <div class="summary-grid" style="grid-template-columns:repeat(3,1fr)">
            <div class="sum-card">
              <div class="sum-num">{s.get("total_accounts",0)}</div>
              <div class="sum-lbl">Total Accounts</div>
            </div>
            <div class="sum-card">
              <div class="sum-num" style="color:var(--ok)">{s.get("normal_accounts",0)}</div>
              <div class="sum-lbl">Normal</div>
            </div>
            <div class="sum-card">
              <div class="sum-num" style="color:var(--muted)">{s.get("closed_accounts",0)}</div>
              <div class="sum-lbl">Closed</div>
            </div>
            <div class="sum-card">
              <div class="sum-num" style="color:var(--danger)">{s.get("delinquent",0)}</div>
              <div class="sum-lbl">Delinquent</div>
            </div>
            <div class="sum-card">
              <div class="sum-num" style="color:var(--warn)">{s.get("writeoff_accounts",0)}</div>
              <div class="sum-lbl">Write-Off</div>
            </div>
            <div class="sum-card">
              <div class="sum-num">{s.get("prev_enquiries",0)}</div>
              <div class="sum-lbl">Prior Enquiries</div>
            </div>
          </div>
        </div>
      </div>

      {emp_html}

      <div class="section-hdr">I. Active Accounts ({len(active)})</div>
      <div style="overflow-x:auto">{_accounts_table(active, False)}</div>

      <div class="section-hdr">II. Closed Accounts ({len(closed) + len(guar_c)})</div>
      <div style="overflow-x:auto">{_accounts_table(closed + guar_c, True)}</div>

    </div>"""
    return panel


@app.get("/download/{uid}")
def download_excel(uid: str):
    """Download the generated Excel file."""
    matches = list(UPLOAD_DIR.glob(f"{uid}_CBC_Summary.xlsx"))
    if not matches:
        raise HTTPException(404, "File not found or expired")

    path    = matches[0]
    content = path.read_bytes()
    fname   = f"CBC_Account_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/extract")
def api_status():
    return {"status": "running", "service": "cbc-portal", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}
