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
    """Upload, extract, preview."""
    uid     = str(uuid.uuid4())[:8]
    suffix  = Path(pdf_file.filename).suffix.lower()
    pdf_path = UPLOAD_DIR / f"{uid}_{pdf_file.filename}"
    content = await pdf_file.read()
    pdf_path.write_bytes(content)

    try:
        from coho_extractor import extract_statement, fill_coho_template, compute_analytics
        data      = extract_statement(str(pdf_path))
        analytics = compute_analytics(data)
        xlsx_path = UPLOAD_DIR / f"{uid}_COHO_Summary.xlsx"
        fill_coho_template(data, str(xlsx_path))
    except Exception as e:
        log.error(f"Extraction failed: {e}")
        return HTMLResponse(_shell("Error",
            f'<div class="alert alert-err">❌ Extraction failed: {e}</div>'))

    header = data["header"]
    txns   = data["transactions"]
    an     = analytics

    # ── Stats ─────────────────────────────────────────────────────────────
    stats_html = f"""
    <div class="stat-grid">
      <div class="stat">
        <div class="stat-num">{len(txns)}</div>
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
        <div class="stat-lbl">Period (Months)</div>
      </div>
      <div class="stat">
        <div class="stat-num">{_fmt(an.get('reversal_amt',0))}</div>
        <div class="stat-lbl">Reversal Entries</div>
      </div>
    </div>"""

    # ── Monthly table ──────────────────────────────────────────────────────
    monthly_html = """
    <table class="month-table">
      <thead>
        <tr>
          <th>#</th><th>Month</th>
          <th>Debit Trx</th><th>Debit Amt (USD)</th>
          <th>Credit Trx</th><th>Credit Amt (USD)</th>
          <th>Avg Closing Bal</th><th>Lowest Bal</th><th>Highest Bal</th>
        </tr>
      </thead>
      <tbody>"""
    for mr in an.get("monthly_rows",[]):
        monthly_html += f"""
        <tr>
          <td>{mr['no']}</td>
          <td>{mr['month'].strftime('%b %Y') if mr.get('month') else ''}</td>
          <td style="color:var(--danger)">{mr['debit_trx']}</td>
          <td style="color:var(--danger)">{mr['debit_amt']:,.2f}</td>
          <td style="color:var(--ok)">{mr['credit_trx']}</td>
          <td style="color:var(--ok)">{mr['credit_amt']:,.2f}</td>
          <td>{mr['avg_bal']:,.2f}</td>
          <td>{mr['lowest_bal']:,.2f}</td>
          <td>{mr['highest_bal']:,.2f}</td>
        </tr>"""
    tot_out = an.get('total_out_amt',0)
    tot_in  = an.get('total_in_amt',0)
    monthly_html += f"""
        <tr class="total-row">
          <td colspan="2">Total</td>
          <td style="color:var(--danger)">{an.get('total_out_trx',0)}</td>
          <td style="color:var(--danger)">{tot_out:,.2f}</td>
          <td style="color:var(--ok)">{an.get('total_in_trx',0)}</td>
          <td style="color:var(--ok)">{tot_in:,.2f}</td>
          <td colspan="3"></td>
        </tr>
      </tbody>
    </table>"""

    # ── Transaction table (first 100) ──────────────────────────────────────
    daily_closing = an.get("daily_closing", {})
    trx_rows = ""
    for i, t in enumerate(txns[:200]):
        is_debit   = t["money_out"] > 0
        is_closing = (t["date"] in daily_closing and
                      abs(t["os_balance"] - daily_closing[t["date"]]) < 0.01)
        row_cls = "closing" if is_closing else ("debit" if is_debit else "")
        trx_rows += f"""
        <tr class="{row_cls}">
          <td style="text-align:center;color:var(--muted)">{i+1}</td>
          <td style="white-space:nowrap">{t['date'].strftime('%d/%m/%Y')}</td>
          <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis"
              title="{t['desc']}">{t['desc'][:70]}</td>
          <td class="amount debit">{f"{t['money_out']:,.2f}" if t['money_out'] > 0 else ''}</td>
          <td class="amount credit">{f"{t['money_in']:,.2f}" if t['money_in'] > 0 else ''}</td>
          <td class="amount">{t['os_balance']:,.2f}</td>
          {'<td class="amount" style="background:#fff3cd;font-weight:700">✓</td>' if is_closing else '<td></td>'}
        </tr>"""

    more_msg = ""
    if len(txns) > 200:
        more_msg = f'<p style="text-align:center;color:var(--muted);padding:12px;font-size:12px">Showing first 200 of {len(txns)} transactions — all {len(txns)} are in the Excel download</p>'

    period_from = header.get("period_from")
    period_to   = header.get("period_to")
    period_str  = ""
    if period_from and period_to:
        period_str = f"{period_from.strftime('%d %b %Y')} – {period_to.strftime('%d %b %Y')}"

    body = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
      <div>
        <h1 style="font-size:20px;font-weight:700">Statement Extracted Successfully</h1>
        <p style="font-size:13px;color:var(--muted);margin-top:3px">{period_str}</p>
      </div>
      <div style="display:flex;gap:10px">
        <a href="/coho/download/{uid}" class="btn btn-primary btn-lg">⬇ Download Excel</a>
        <a href="/coho/" class="btn btn-ghost">Upload Another</a>
      </div>
    </div>

    <div class="step-indicator">
      <div class="step done"><div class="step-dot">✓</div>
        <div class="step-label">Upload PDF</div><div class="step-line"></div></div>
      <div class="step done"><div class="step-dot">✓</div>
        <div class="step-label">Extract & Analyse</div><div class="step-line"></div></div>
      <div class="step active"><div class="step-dot">3</div>
        <div class="step-label">Download Excel</div></div>
    </div>

    <div class="card">
      <div class="card-title">Account Information</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:12px">
        <div><span style="color:var(--muted)">Bank: </span><strong>{header.get('bank','')}</strong></div>
        <div><span style="color:var(--muted)">Account Holder: </span><strong>{header.get('holder_name','')}</strong></div>
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
      <div class="card-title">Transactions Preview
        <span style="font-weight:400;color:var(--muted);margin-left:8px">
          🟡 highlighted = daily closing balance
        </span>
      </div>
      <div style="overflow-x:auto">
        <table class="trx-table">
          <thead>
            <tr>
              <th>#</th><th>Date</th><th>Transaction Details</th>
              <th>Debit (USD)</th><th>Credit (USD)</th>
              <th>O/S Balance</th><th>Closing</th>
            </tr>
          </thead>
          <tbody>
            {trx_rows}
          </tbody>
        </table>
      </div>
      {more_msg}
    </div>

    <div style="text-align:center;padding:20px">
      <a href="/coho/download/{uid}" class="btn btn-primary btn-lg">
        ⬇ Download Filled COHO Excel Template
      </a>
      <p style="font-size:12px;color:var(--muted);margin-top:10px">
        Uniform Conduct of Account Summary — ready for credit analysis
      </p>
    </div>"""

    return HTMLResponse(_shell("Review", body))


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
