"""
coho_portal.py — Conduct of Account (COHO) Statement Analysis Portal
Web interface for uploading bank statement PDFs, extracting all transactions,
computing COHO analytics, previewing results, and downloading the filled Excel.

Runs on port 8008. Access at /coho/ via Nginx.
Integrated into the existing loan officer portal system.
"""

import os
import io
import json
import uuid
import logging
from datetime import datetime, date
from pathlib import Path

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("coho_portal")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads")) / "coho"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="COHO Bank Statement Portal", version="1.0.0")

CSS = """
:root{
  --nav:#1a2744;--accent:#2563eb;--accent-l:#eff6ff;
  --ok:#059669;--ok-bg:#ecfdf5;--warn:#d97706;--warn-bg:#fffbeb;
  --danger:#dc2626;--danger-bg:#fef2f2;
  --surface:#f8fafc;--card:#fff;--border:#e2e8f0;
  --text:#0f172a;--muted:#64748b;
  --blue-dark:#1F4E79;--blue-light:#D6E4F0;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);font-size:14px}
.topbar{background:var(--nav);color:#fff;height:56px;display:flex;
        align-items:center;padding:0 24px;justify-content:space-between;
        box-shadow:0 2px 8px rgba(0,0,0,.2);position:sticky;top:0;z-index:100}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;background:var(--blue-dark);border-radius:7px;
            display:flex;align-items:center;justify-content:center;
            font-size:11px;font-weight:800;color:#fff}
.brand-title{font-size:14px;font-weight:700;color:#f1f5f9}
.brand-sub{font-size:11px;color:#94a3b8}
.nav-link{color:#94a3b8;font-size:12px;padding:6px 12px;border-radius:6px;
          text-decoration:none;transition:all .15s}
.nav-link:hover{background:rgba(255,255,255,.1);color:#fff}
.nav-links{display:flex;gap:6px}
.content{max-width:1280px;margin:0 auto;padding:24px 20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:20px 24px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.card-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;
            letter-spacing:.6px;margin-bottom:14px;padding-bottom:8px;
            border-bottom:1px solid var(--border)}
.upload-zone{border:2px dashed #cbd5e1;border-radius:12px;padding:40px;
             text-align:center;cursor:pointer;transition:all .2s;background:#f8fafc}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:var(--accent-l)}
.upload-icon{font-size:36px;margin-bottom:12px}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 20px;
     border-radius:8px;font-size:13px;font-weight:500;cursor:pointer;
     border:none;transition:all .15s;text-decoration:none}
.btn-primary{background:var(--blue-dark);color:#fff}
.btn-primary:hover{background:#163a5c}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface)}
.btn-lg{padding:13px 28px;font-size:14px;font-weight:600}
.progress{height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;margin:12px 0}
.progress-bar{height:100%;background:var(--accent);border-radius:3px;
              transition:width .3s ease;width:0%}
.alert{padding:12px 16px;border-radius:8px;font-size:13px;margin-bottom:14px;
       display:flex;align-items:flex-start;gap:10px}
.alert-ok{background:var(--ok-bg);border:1px solid #6ee7b7;color:#065f46}
.alert-err{background:var(--danger-bg);border:1px solid #fca5a5;color:#991b1b}
.alert-info{background:var(--accent-l);border:1px solid #93c5fd;color:#1e40af}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:10px;
      padding:14px 16px;text-align:center}
.stat-num{font-size:22px;font-weight:700;margin-bottom:3px}
.stat-lbl{font-size:11px;color:var(--muted);font-weight:500}
.stat-credit{color:var(--ok)}
.stat-debit{color:var(--danger)}
.step-indicator{display:flex;align-items:center;gap:0;margin-bottom:24px}
.step{display:flex;align-items:center;flex:1}
.step-dot{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
          justify-content:center;font-size:12px;font-weight:700;flex-shrink:0;
          border:2px solid var(--border);background:var(--card);color:var(--muted)}
.step.done .step-dot{background:var(--ok);border-color:var(--ok);color:#fff}
.step.active .step-dot{background:var(--accent);border-color:var(--accent);color:#fff}
.step-label{font-size:11px;color:var(--muted);margin-left:8px;font-weight:500}
.step.active .step-label{color:var(--accent);font-weight:700}
.step.done .step-label{color:var(--ok)}
.step-line{flex:1;height:2px;background:var(--border);margin:0 6px}
.step.done .step-line{background:var(--ok)}
.trx-table{width:100%;border-collapse:collapse;font-size:11px}
.trx-table th{background:var(--blue-light);color:var(--blue-dark);
              padding:7px 8px;text-align:left;font-size:10px;font-weight:700;
              text-transform:uppercase;letter-spacing:.3px;border-bottom:2px solid #bdd7ee}
.trx-table td{padding:6px 8px;border-bottom:1px solid #f1f5f9;vertical-align:middle}
.trx-table tr:hover td{background:#f8fafc}
.trx-table tr.debit td{background:#fff5f5}
.trx-table tr.closing td{background:#fff3cd}
.amount{font-family:monospace;font-size:11px;font-weight:600}
.credit{color:var(--ok)}
.debit{color:var(--danger)}
.month-table{width:100%;border-collapse:collapse;font-size:12px}
.month-table th{background:var(--blue-light);color:var(--blue-dark);padding:8px 10px;
                text-align:center;font-size:11px;font-weight:700;border:1px solid #bdd7ee}
.month-table td{padding:7px 10px;border:1px solid #e2e8f0;text-align:right}
.month-table td:first-child,.month-table td:nth-child(2){text-align:center}
.month-table tr:nth-child(even) td{background:#f8fafc}
.month-table .total-row td{background:#fffde7;font-weight:700}
"""

HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>COHO Statement Analyser — {title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="brand-icon">COHO</div>
      <div>
        <div class="brand-title">Conduct of Account Analyser</div>
        <div class="brand-sub">Bank Statement PDF → Excel Summary</div>
      </div>
    </div>
    <div class="nav-links">
      <a href="/coho/" class="nav-link">Upload</a>
      <a href="/cbc/" class="nav-link">CBC Portal</a>
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

def _fmt(v, currency="USD"):
    try:
        f = float(v)
        return f"{currency} {f:,.2f}"
    except: return str(v)


@app.get("/", response_class=HTMLResponse)
def index():
    body = """
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
      <div>
        <h1 style="font-size:20px;font-weight:700">Bank Statement Analyser</h1>
        <p style="font-size:13px;color:var(--muted);margin-top:3px">
          Upload a bank statement PDF → extract all transactions → download the COHO Excel template
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
        <div class="step-label">Extract & Analyse</div>
        <div class="step-line"></div>
      </div>
      <div class="step">
        <div class="step-dot">3</div>
        <div class="step-label">Download Excel</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Upload Bank Statement PDF</div>
      <div class="alert alert-info">
        <span>ℹ</span>
        <div>Supports ABA Bank statements and other Cambodian bank PDF statements.
        All transactions are extracted automatically — no manual keying required.</div>
      </div>
      <form id="upload-form" action="/coho/extract" method="post"
            enctype="multipart/form-data">
        <div class="upload-zone" id="drop-zone"
             onclick="document.getElementById('pdf-input').click()">
          <div class="upload-icon">📄</div>
          <div style="font-size:16px;font-weight:600;margin-bottom:6px">
            Drop bank statement PDF here or click to browse
          </div>
          <div style="font-size:13px;color:var(--muted)">
            PDF or image format · Max 30MB
          </div>
          <input type="file" id="pdf-input" name="pdf_file"
                 accept=".pdf,.jpg,.jpeg,.png" style="display:none"
                 onchange="handleFile(this)">
        </div>
        <div id="file-info" style="display:none;margin-top:12px">
          <div class="alert alert-ok" id="file-name"></div>
          <div class="progress"><div class="progress-bar" id="pbar"></div></div>
        </div>
        <div style="margin-top:16px;display:flex;gap:10px">
          <button type="submit" class="btn btn-primary btn-lg"
                  id="submit-btn" disabled onclick="startUpload()">
            Extract & Generate Excel
          </button>
          <button type="button" class="btn btn-ghost" onclick="resetForm()">Clear</button>
        </div>
      </form>
    </div>

    <div class="card">
      <div class="card-title">What gets extracted & calculated</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
        <div style="padding:14px;background:#eff6ff;border-radius:8px;border:1px solid #bfdbfe">
          <div style="font-size:20px;margin-bottom:8px">📋</div>
          <div style="font-weight:600;margin-bottom:4px;font-size:13px">All Transactions</div>
          <div style="font-size:12px;color:var(--muted)">
            Date, description, debit, credit, running balance — every row from every page
          </div>
        </div>
        <div style="padding:14px;background:#f0fdf4;border-radius:8px;border:1px solid #bbf7d0">
          <div style="font-size:20px;margin-bottom:8px">📊</div>
          <div style="font-weight:600;margin-bottom:4px;font-size:13px">COHO Summary A</div>
          <div style="font-size:12px;color:var(--muted)">
            Total/average transactions, turnover Dr/Cr, highest/lowest/average
            closing balance, reversal entries
          </div>
        </div>
        <div style="padding:14px;background:#fef3c7;border-radius:8px;border:1px solid #fcd34d">
          <div style="font-size:20px;margin-bottom:8px">📅</div>
          <div style="font-weight:600;margin-bottom:4px;font-size:13px">COHO Summary B</div>
          <div style="font-size:12px;color:var(--muted)">
            Month-by-month breakdown: debit/credit count & amount, average closing balance,
            lowest/highest balance per month
          </div>
        </div>
      </div>
    </div>"""

    scripts = """<script>
const zone = document.getElementById('drop-zone');
const inp  = document.getElementById('pdf-input');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.classList.remove('drag');
  if (e.dataTransfer.files.length) { inp.files = e.dataTransfer.files; handleFile(inp); }
});
function handleFile(i) {
  const f = i.files[0]; if (!f) return;
  document.getElementById('file-info').style.display = 'block';
  document.getElementById('file-name').innerHTML =
    '✓ Selected: <strong>' + f.name + '</strong> (' + (f.size/1024/1024).toFixed(2) + ' MB)';
  document.getElementById('submit-btn').disabled = false;
}
function resetForm() {
  document.getElementById('upload-form').reset();
  document.getElementById('file-info').style.display = 'none';
  document.getElementById('submit-btn').disabled = true;
}
function startUpload() {
  const btn = document.getElementById('submit-btn');
  btn.textContent = 'Extracting...'; btn.disabled = true;
  let p = 0; const bar = document.getElementById('pbar');
  setInterval(() => { p = Math.min(p + Math.random()*12, 88); bar.style.width = p + '%'; }, 400);
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

    # Store job metadata
    job_file = UPLOAD_DIR / f"{uid}_job.json"
    job_file.write_text(json.dumps({
        "uid":      uid,
        "filename": pdf_file.filename,
        "pdf_path": str(pdf_path),
        "status":   "queued",
        "size_mb":  round(len(content)/1024/1024, 2),
    }))

    # Redirect to progress page
    from fastapi.responses import RedirectResponse
    return RedirectResponse(f"/coho/progress/{uid}", status_code=303)


@app.get("/progress/{uid}", response_class=HTMLResponse)
def progress_page(uid: str):
    """Progress page - starts extraction and polls for updates."""
    body = f"""
    <div style="max-width:700px;margin:0 auto">
      <h1 style="font-size:20px;font-weight:700;margin-bottom:6px">Processing Statement</h1>
      <p style="font-size:13px;color:var(--muted);margin-bottom:20px">
        Extracting transactions from your bank statement PDF...
      </p>
      <div class="card">
        <div class="card-title">Extraction Progress</div>
        <div class="progress" style="height:10px;margin-bottom:16px">
          <div class="progress-bar" id="pbar" style="width:5%;transition:width .4s"></div>
        </div>
        <div id="status-text" style="font-size:13px;font-weight:600;color:var(--accent);margin-bottom:12px">
          Starting...
        </div>
        <div id="log-box" style="
          background:#0f172a;color:#e2e8f0;font-family:monospace;font-size:12px;
          padding:16px;border-radius:8px;height:300px;overflow-y:auto;
          line-height:1.7;white-space:pre-wrap;"></div>
      </div>
    </div>

    <script>
    const uid    = '{uid}';
    const logBox = document.getElementById('log-box');
    const pbar   = document.getElementById('pbar');
    const statT  = document.getElementById('status-text');
    let   from_  = 0;

    function appendLog(msg, color) {{
      const s = document.createElement('span');
      s.style.color = color || '#e2e8f0';
      s.textContent = msg + '\\n';
      logBox.appendChild(s);
      logBox.scrollTop = logBox.scrollHeight;
    }}

    // Kick off extraction in background
    fetch('/coho/run/' + uid).catch(() => {{}});

    // Poll every 800ms
    const timer = setInterval(async () => {{
      try {{
        const r = await fetch('/coho/poll/' + uid + '?from=' + from_);
        if (!r.ok) return;
        const d = await r.json();

        (d.logs || []).forEach(l => {{
          appendLog(l.msg, l.color);
          from_++;
        }});

        if (d.pct)   pbar.style.width = d.pct + '%';
        if (d.label) statT.textContent = d.label;

        if (d.done) {{
          clearInterval(timer);
          pbar.style.width  = '100%';
          statT.textContent = '✅ Complete — opening results...';
          appendLog('\\n✅ Done!', '#4ade80');
          setTimeout(() => window.location.href = '/coho/result/' + uid, 800);
        }}
        if (d.error) {{
          clearInterval(timer);
          pbar.style.width  = '0%';
          statT.textContent = '❌ Failed';
          appendLog('\\n❌ ' + d.error, '#f87171');
        }}
      }} catch(e) {{ /* retry */ }}
    }}, 800);
    </script>"""
    return HTMLResponse(_shell("Processing", body))


@app.get("/run/{uid}")
async def run_extraction(uid: str):
    """Background endpoint — runs extraction and writes log file."""
    import asyncio, concurrent.futures

    job_file = UPLOAD_DIR / f"{uid}_job.json"
    if not job_file.exists():
        return JSONResponse({"error": "job not found"})

    job      = json.loads(job_file.read_text())
    pdf_path = job["pdf_path"]
    log_file = UPLOAD_DIR / f"{uid}_log.json"
    err_file = UPLOAD_DIR / f"{uid}_error.txt"

    logs = []

    def append(msg, color="#e2e8f0", pct=None, label=None):
        entry = {"msg": msg, "color": color}
        if pct   is not None: entry["pct"]   = pct
        if label is not None: entry["label"] = label
        logs.append(entry)
        log_file.write_text(json.dumps(logs))

    def run():
        try:
            from coho_extractor import (
                BankStatementParser, compute_analytics, fill_coho_template
            )
            append(f"📄 File: {job['filename']} ({job['size_mb']} MB)", "#94a3b8", 5, "Opening PDF...")
            append(f"⏱  Started: {datetime.now().strftime('%H:%M:%S')}", "#94a3b8")
            append("")

            parser  = BankStatementParser(pdf_path)
            n_pages = len(parser.pages)
            append(f"🔍 Pages found: {n_pages}", "#93c5fd", 15, "Reading pages...")

            header = parser._parse_header()
            append(f"   Bank:    {header.get('bank','?')}", "#e2e8f0", 25, "Extracting header...")
            append(f"   Holder:  {header.get('holder_name','?')}")
            append(f"   Account: {header.get('account_no','?')}")
            append(f"   Period:  {header.get('period_from','?')} → {header.get('period_to','?')}")

            append("", "#e2e8f0", 40, "Parsing transactions...")
            txns = parser._parse_transactions()
            append(f"💳 Transactions found: {len(txns)}", "#4ade80", 55, "Fixing balances...")

            txns = parser._fix_balances(header.get("opening_balance", 0), txns)
            data = {
                "header": header, "transactions": txns,
                "parsed_at": datetime.now().isoformat(),
                "source_file": pdf_path,
            }

            append("", "#e2e8f0", 65, "Computing analytics...")
            an = compute_analytics(data)
            append(f"   Total In:  ${an.get('total_in_amt',0):,.2f} ({an.get('total_in_trx',0)} trx)", "#4ade80")
            append(f"   Total Out: ${an.get('total_out_amt',0):,.2f} ({an.get('total_out_trx',0)} trx)", "#f87171")
            append(f"   Avg Bal:   ${an.get('avg_closing_bal',0):,.2f}")
            append(f"   Highest:   ${an.get('highest_balance',0):,.2f}")
            append(f"   Months:    {an.get('period_months',0)}")

            append("", "#e2e8f0", 80, "Generating Excel...")
            xlsx_path = str(UPLOAD_DIR / f"{uid}_COHO_Summary.xlsx")
            fill_coho_template(data, xlsx_path)
            append("📊 Excel generated!", "#4ade80", 92, "Saving result...")

            # Save result JSON
            result_file = UPLOAD_DIR / f"{uid}_result.json"
            result_file.write_text(json.dumps({
                "uid": uid,
                "header":  {k: str(v) for k, v in header.items()},
                "analytics": {
                    k: v for k, v in an.items()
                    if k not in ("monthly_rows","daily_closing")
                },
                "monthly_rows": [
                    {k2: str(v2) if hasattr(v2,"strftime") else v2
                     for k2, v2 in mr.items()}
                    for mr in an.get("monthly_rows", [])
                ],
                "total_transactions": len(txns),
                "transactions": [
                    {
                        "date":       t["date"].isoformat(),
                        "desc":       t["desc"],
                        "money_in":   t["money_in"],
                        "money_out":  t["money_out"],
                        "os_balance": t["os_balance"],
                        "is_closing": (
                            t["date"] in an.get("daily_closing", {}) and
                            abs(t["os_balance"] -
                                an["daily_closing"].get(t["date"], 99999)) < 0.01
                        ),
                    }
                    for t in txns
                ],
            }))
            append("✅ Complete!", "#4ade80", 100, "Done!")

        except Exception as e:
            import traceback
            err_file.write_text(str(e))
            append(f"❌ Error: {e}", "#f87171")
            append(traceback.format_exc(), "#f87171")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run)
    return JSONResponse({"started": True})


@app.get("/poll/{uid}")
def poll(uid: str, from_: int = 0):
    """Poll extraction progress — returns new log lines since last poll."""
    log_file  = UPLOAD_DIR / f"{uid}_log.json"
    done_file = UPLOAD_DIR / f"{uid}_result.json"
    err_file  = UPLOAD_DIR / f"{uid}_error.txt"

    logs = []
    pct  = 5
    label = "Processing..."

    if log_file.exists():
        try:
            all_logs = json.loads(log_file.read_text())
            logs     = all_logs[from_:]
            if all_logs:
                last  = all_logs[-1]
                pct   = last.get("pct", pct)
                label = last.get("label", label)
        except: pass

    return JSONResponse({
        "logs":  logs,
        "pct":   pct,
        "label": label,
        "done":  done_file.exists(),
        "error": err_file.read_text() if err_file.exists() else None,
    })








@app.get("/result/{uid}", response_class=HTMLResponse)
def result_page(uid: str):
    """Show results after extraction completes."""
    result_file = UPLOAD_DIR / f"{uid}_result.json"
    if not result_file.exists():
        return HTMLResponse(_shell("Error",
            '<div class="alert alert-err">Result not found. Please upload again.</div>'))

    result = json.loads(result_file.read_text())
    header = result.get("header", {})
    an     = result.get("analytics", {})
    txns   = result.get("transactions", [])
    monthly_rows = result.get("monthly_rows", [])

    # ── Stats ──────────────────────────────────────────────────────────────
    stats_html = f"""
    <div class="stat-grid">
      <div class="stat">
        <div class="stat-num">{result.get('total_transactions',0)}</div>
        <div class="stat-lbl">Total Transactions</div>
      </div>
      <div class="stat">
        <div class="stat-num stat-credit">{_fmt(an.get('total_in_amt',0))}</div>
        <div class="stat-lbl">Total Money In ({an.get('total_in_trx',0)} trx)</div>
      </div>
      <div class="stat">
        <div class="stat-num stat-debit">{_fmt(an.get('total_out_amt',0))}</div>
        <div class="stat-lbl">Total Money Out ({an.get('total_out_trx',0)} trx)</div>
      </div>
      <div class="stat">
        <div class="stat-num">{_fmt(an.get('avg_closing_bal',0))}</div>
        <div class="stat-lbl">Avg Closing Balance</div>
      </div>
      <div class="stat">
        <div class="stat-num">{_fmt(an.get('highest_balance',0))}</div>
        <div class="stat-lbl">Highest Balance</div>
      </div>
      <div class="stat">
        <div class="stat-num">{_fmt(an.get('lowest_balance',0))}</div>
        <div class="stat-lbl">Lowest Balance</div>
      </div>
      <div class="stat">
        <div class="stat-num">{an.get('period_months',0)}</div>
        <div class="stat-lbl">Months</div>
      </div>
      <div class="stat">
        <div class="stat-num">{_fmt(an.get('reversal_amt',0))}</div>
        <div class="stat-lbl">Reversal Entries</div>
      </div>
    </div>"""

    # ── Monthly table ──────────────────────────────────────────────────────
    monthly_html = """<table class="month-table">
      <thead><tr>
        <th>#</th><th>Month</th>
        <th>Debit Trx</th><th>Debit Amt</th>
        <th>Credit Trx</th><th>Credit Amt</th>
        <th>Avg Closing</th><th>Lowest</th><th>Highest</th>
      </tr></thead><tbody>"""
    for mr in monthly_rows:
        monthly_html += f"""<tr>
          <td>{mr.get('no','')}</td>
          <td>{mr.get('month','')[:7]}</td>
          <td style="color:var(--danger)">{mr.get('debit_trx',0)}</td>
          <td style="color:var(--danger)">{float(mr.get('debit_amt',0)):,.2f}</td>
          <td style="color:var(--ok)">{mr.get('credit_trx',0)}</td>
          <td style="color:var(--ok)">{float(mr.get('credit_amt',0)):,.2f}</td>
          <td>{float(mr.get('avg_bal',0)):,.2f}</td>
          <td>{float(mr.get('lowest_bal',0)):,.2f}</td>
          <td>{float(mr.get('highest_bal',0)):,.2f}</td>
        </tr>"""
    monthly_html += f"""<tr class="total-row">
      <td colspan="2">Total</td>
      <td style="color:var(--danger)">{an.get('total_out_trx',0)}</td>
      <td style="color:var(--danger)">{float(an.get('total_out_amt',0)):,.2f}</td>
      <td style="color:var(--ok)">{an.get('total_in_trx',0)}</td>
      <td style="color:var(--ok)">{float(an.get('total_in_amt',0)):,.2f}</td>
      <td colspan="3"></td>
    </tr></tbody></table>"""

    # ── Transaction preview ────────────────────────────────────────────────
    trx_rows = ""
    for i, t in enumerate(txns[:300]):
        is_debit   = float(t.get("money_out",0)) > 0
        is_closing = t.get("is_closing", False)
        row_cls    = "closing" if is_closing else ("debit" if is_debit else "")
        trx_rows += f"""<tr class="{row_cls}">
          <td style="text-align:center;color:var(--muted)">{i+1}</td>
          <td style="white-space:nowrap">{t['date'][:10]}</td>
          <td title="{t['desc']}">{t['desc'][:70]}</td>
          <td class="amount debit">{f"{float(t['money_out']):,.2f}" if float(t.get('money_out',0)) > 0 else ''}</td>
          <td class="amount credit">{f"{float(t['money_in']):,.2f}" if float(t.get('money_in',0)) > 0 else ''}</td>
          <td class="amount">{float(t.get('os_balance',0)):,.2f}</td>
          <td>{"🟡" if is_closing else ""}</td>
        </tr>"""

    more_msg = f'<p style="text-align:center;color:var(--muted);padding:10px;font-size:12px">Showing first 300 of {len(txns)} transactions</p>' \
               if len(txns) > 300 else ""

    period_str = f"{header.get('period_from','')[:10]} – {header.get('period_to','')[:10]}"

    body = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
      <div>
        <h1 style="font-size:20px;font-weight:700">✅ Extraction Complete</h1>
        <p style="font-size:13px;color:var(--muted);margin-top:3px">{period_str}</p>
      </div>
      <div style="display:flex;gap:10px">
        <a href="/coho/download/{uid}" class="btn btn-primary btn-lg">⬇ Download Excel</a>
        <a href="/coho/" class="btn btn-ghost">Upload Another</a>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Account Information</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:12px">
        <div><span style="color:var(--muted)">Bank: </span><strong>{header.get('bank','')}</strong></div>
        <div><span style="color:var(--muted)">Holder: </span><strong>{header.get('holder_name','')}</strong></div>
        <div><span style="color:var(--muted)">Account No: </span><strong>{header.get('account_no','')}</strong></div>
        <div><span style="color:var(--muted)">Currency: </span><strong>{header.get('currency','')}</strong></div>
        <div><span style="color:var(--muted)">Opening Balance: </span><strong>{_fmt(header.get('opening_balance',0))}</strong></div>
        <div><span style="color:var(--muted)">Ending Balance: </span><strong>{_fmt(header.get('ending_balance',0))}</strong></div>
        <div><span style="color:var(--muted)">Period: </span><strong>{period_str}</strong></div>
        <div><span style="color:var(--muted)">Months: </span><strong>{an.get('period_months',0)}</strong></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">COHO Summary — Key Statistics</div>
      {stats_html}
    </div>

    <div class="card">
      <div class="card-title">Monthly Breakdown (Section B)</div>
      <div style="overflow-x:auto">{monthly_html}</div>
    </div>

    <div class="card">
      <div class="card-title">Transactions
        <span style="font-weight:400;color:var(--muted);margin-left:8px;font-size:11px">
          🟡 = daily closing balance
        </span>
      </div>
      <div style="overflow-x:auto">
        <table class="trx-table">
          <thead><tr>
            <th>#</th><th>Date</th><th>Description</th>
            <th>Debit (USD)</th><th>Credit (USD)</th>
            <th>Balance</th><th></th>
          </tr></thead>
          <tbody>{trx_rows}</tbody>
        </table>
      </div>
      {more_msg}
    </div>

    <div style="text-align:center;padding:20px">
      <a href="/coho/download/{uid}" class="btn btn-primary btn-lg">
        ⬇ Download Filled COHO Excel Template
      </a>
    </div>"""

    return HTMLResponse(_shell("Results", body))


@app.get("/download/{uid}")
def download(uid: str):
    matches = list(UPLOAD_DIR.glob(f"{uid}_COHO_Summary.xlsx"))
    if not matches:
        raise HTTPException(404, "File not found")
    path    = matches[0]
    content = path.read_bytes()
    fname   = f"COHO_Statement_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "coho-portal"}
