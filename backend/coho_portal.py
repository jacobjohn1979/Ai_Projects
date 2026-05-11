"""
coho_portal.py — Conduct of Account Bank Statement Portal
Upload PDF → extract transactions → download COHO Excel template
Port: 8008  Access: /coho/
"""
import os, io, json, uuid, logging, threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()
log        = logging.getLogger("coho_portal")
BASE_DIR   = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
UPLOAD_DIR = BASE_DIR / "coho"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="COHO Portal")

CSS = """
:root{--nav:#1a2744;--accent:#2563eb;--ok:#059669;--ok-bg:#ecfdf5;
  --danger:#dc2626;--surface:#f8fafc;--card:#fff;--border:#e2e8f0;
  --text:#0f172a;--muted:#64748b;--blue-dark:#1F4E79;--blue-light:#D6E4F0;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);font-size:14px}
.topbar{background:var(--nav);color:#fff;height:56px;display:flex;
        align-items:center;padding:0 24px;justify-content:space-between;
        box-shadow:0 2px 8px rgba(0,0,0,.2);position:sticky;top:0;z-index:100}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;background:var(--blue-dark);border-radius:7px;
            display:flex;align-items:center;justify-content:center;font-size:11px;
            font-weight:800;color:#fff}
.brand-title{font-size:14px;font-weight:700;color:#f1f5f9}
.brand-sub{font-size:11px;color:#94a3b8}
.nav-link{color:#94a3b8;font-size:12px;padding:6px 12px;border-radius:6px;text-decoration:none}
.nav-link:hover{background:rgba(255,255,255,.1);color:#fff}
.nav-links{display:flex;gap:6px}
.content{max-width:1100px;margin:0 auto;padding:24px 20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:20px 24px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.card-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;
            letter-spacing:.6px;margin-bottom:14px;padding-bottom:8px;
            border-bottom:1px solid var(--border)}
.upload-zone{border:2px dashed #cbd5e1;border-radius:12px;padding:40px;
             text-align:center;cursor:pointer;transition:all .2s;background:#f8fafc}
.upload-zone:hover{border-color:var(--accent);background:#eff6ff}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 20px;
     border-radius:8px;font-size:13px;font-weight:500;cursor:pointer;
     border:none;transition:all .15s;text-decoration:none}
.btn-primary{background:var(--blue-dark);color:#fff}
.btn-primary:hover{background:#163a5c}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-lg{padding:13px 28px;font-size:14px;font-weight:600}
.progress{height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;margin:12px 0}
.progress-bar{height:100%;background:var(--accent);border-radius:4px;transition:width .5s ease;width:0%}
.alert-ok{background:var(--ok-bg);border:1px solid #6ee7b7;color:#065f46;
          padding:10px 14px;border-radius:8px;font-size:13px;margin-top:10px}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;text-align:center}
.stat-num{font-size:20px;font-weight:700;margin-bottom:3px}
.stat-lbl{font-size:11px;color:var(--muted);font-weight:500}
.trx-table{width:100%;border-collapse:collapse;font-size:11px}
.trx-table th{background:var(--blue-light);color:var(--blue-dark);padding:7px 8px;
              text-align:left;font-size:10px;font-weight:700;border-bottom:2px solid #bdd7ee}
.trx-table td{padding:6px 8px;border-bottom:1px solid #f1f5f9}
.trx-table tr.debit td{background:#fff5f5}
.trx-table tr.closing td{background:#fffde7}
.month-table{width:100%;border-collapse:collapse;font-size:12px}
.month-table th{background:var(--blue-light);color:var(--blue-dark);padding:8px;
                text-align:center;font-weight:700;border:1px solid #bdd7ee}
.month-table td{padding:7px 8px;border:1px solid #e2e8f0;text-align:right}
.month-table td:nth-child(1),.month-table td:nth-child(2){text-align:center}
.month-table tr:nth-child(even) td{background:#f8fafc}
.month-table .total-row td{background:#fffde7;font-weight:700}
"""

def _sym(currency):
    """Return currency symbol and formatter based on currency code."""
    if str(currency).upper() == "KHR":
        return "KHR", lambda v: "{:,.0f}".format(float(v))  # no decimals for KHR
    return "$", lambda v: "{:,.2f}".format(float(v))


def page(title, body):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>COHO - {title}</title>
<style>{CSS}</style></head>
<body>
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">COHO</div>
    <div><div class="brand-title">Conduct of Account Analyser</div>
         <div class="brand-sub">Bank Statement PDF to Excel</div></div>
  </div>
  <div class="nav-links">
    <a href="/coho/" class="nav-link">Upload</a>
    <a href="/cbc/" class="nav-link">CBC Portal</a>
    <a href="/loan/" class="nav-link">Loan Portal</a>
  </div>
</div>
<div class="content">{body}</div>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    body = """
    <h1 style="font-size:20px;font-weight:700;margin-bottom:20px">Bank Statement Analyser</h1>
    <div class="card">
      <div class="card-title">Upload Bank Statement PDF</div>
      <div class="upload-zone" id="zone" onclick="document.getElementById('inp').click()">
        <div style="font-size:32px;margin-bottom:10px">&#128196;</div>
        <div style="font-size:16px;font-weight:600;margin-bottom:6px">Drop PDF here or click to browse</div>
        <div style="font-size:13px;color:#64748b">ABA Bank and other Cambodian banks - Max 30MB</div>
        <input type="file" id="inp" accept=".pdf" style="display:none" onchange="selectFile(this)">
      </div>
      <div id="info" style="display:none;margin-top:12px">
        <div class="alert-ok" id="fname"></div>
        <div class="progress"><div class="progress-bar" id="pbar"></div></div>
      </div>
      <div style="margin-top:16px;display:flex;gap:10px">
        <button id="btn" class="btn btn-primary btn-lg" disabled onclick="doUpload()">
          Extract and Generate Excel
        </button>
        <button class="btn btn-ghost" onclick="clearForm()">Clear</button>
      </div>
    </div>
    <script>
    function selectFile(inp) {
      var f = inp.files[0];
      if (!f) return;
      document.getElementById('info').style.display = 'block';
      document.getElementById('fname').textContent = 'Selected: ' + f.name + ' (' + (f.size/1024/1024).toFixed(2) + ' MB)';
      document.getElementById('btn').disabled = false;
    }
    function clearForm() {
      document.getElementById('inp').value = '';
      document.getElementById('info').style.display = 'none';
      document.getElementById('btn').disabled = true;
    }
    function doUpload() {
      var inp = document.getElementById('inp');
      if (!inp.files.length) return;
      var btn = document.getElementById('btn');
      btn.textContent = 'Uploading...';
      btn.disabled = true;
      document.getElementById('pbar').style.width = '40%';
      var fd = new FormData();
      fd.append('pdf_file', inp.files[0]);
      fetch('/coho/extract', {method: 'POST', body: fd})
        .then(function(r) { return r.json(); })
        .then(function(d) {
          if (d.uid) {
            window.location.href = '/coho/progress/' + d.uid;
          } else {
            btn.textContent = 'Error: ' + (d.error || 'unknown');
            btn.disabled = false;
          }
        })
        .catch(function(e) {
          btn.textContent = 'Failed: ' + e;
          btn.disabled = false;
        });
    }
    </script>"""
    return HTMLResponse(page("Upload", body))


@app.post("/extract")
async def extract(pdf_file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uid      = str(uuid.uuid4())[:8]
    pdf_path = UPLOAD_DIR / (uid + "_" + pdf_file.filename)
    raw      = await pdf_file.read()
    pdf_path.write_bytes(raw)
    (UPLOAD_DIR / (uid + "_job.json")).write_text(json.dumps({
        "uid": uid, "filename": pdf_file.filename,
        "pdf_path": str(pdf_path),
        "size_mb": round(len(raw) / 1024 / 1024, 2),
    }))
    return JSONResponse({"uid": uid})


@app.get("/progress/{uid}", response_class=HTMLResponse)
def progress(uid: str):
    _start_extraction(uid)
    uid_js = json.dumps(uid)
    body = """
    <div style="max-width:700px;margin:0 auto">
      <h1 style="font-size:20px;font-weight:700;margin-bottom:6px">Processing Statement</h1>
      <p style="font-size:13px;color:#64748b;margin-bottom:20px">Extracting transactions from your PDF...</p>
      <div class="card">
        <div class="card-title">Extraction Progress</div>
        <div class="progress" style="height:10px;margin-bottom:12px">
          <div class="progress-bar" id="pbar" style="width:5%"></div>
        </div>
        <div id="status" style="font-size:13px;font-weight:600;color:#2563eb;margin-bottom:12px">Starting...</div>
        <div id="log" style="background:#0f172a;color:#e2e8f0;font-family:monospace;font-size:12px;
             padding:16px;border-radius:8px;height:300px;overflow-y:auto;line-height:1.7;white-space:pre-wrap"></div>
      </div>
    </div>
    <script>
    var uid   = """ + uid_js + """;
    var from_ = 0;
    var logEl = document.getElementById('log');
    var pbar  = document.getElementById('pbar');
    var stat  = document.getElementById('status');
    function addLog(msg, color) {
      var s = document.createElement('span');
      s.style.color = color || '#e2e8f0';
      s.textContent = msg + '\\n';
      logEl.appendChild(s);
      logEl.scrollTop = logEl.scrollHeight;
    }
    var timer = setInterval(function() {
      fetch('/coho/poll/' + uid + '?from_=' + from_)
        .then(function(r) { return r.json(); })
        .then(function(d) {
          var logs = d.logs || [];
          for (var i = 0; i < logs.length; i++) {
            addLog(logs[i].msg, logs[i].color);
            from_++;
          }
          if (d.pct)   pbar.style.width = d.pct + '%';
          if (d.label) stat.textContent = d.label;
          if (d.done) {
            clearInterval(timer);
            pbar.style.width = '100%';
            stat.textContent = 'Complete - opening results...';
            addLog('\\nDone!', '#4ade80');
            setTimeout(function() { window.location.href = '/coho/result/' + uid; }, 1000);
          }
          if (d.error) {
            clearInterval(timer);
            stat.textContent = 'Failed';
            addLog('\\nError: ' + d.error, '#f87171');
          }
        })
        .catch(function() {});
    }, 1000);
    </script>"""
    return HTMLResponse(page("Processing", body))


def _start_extraction(uid):
    job_file = UPLOAD_DIR / (uid + "_job.json")
    if not job_file.exists():
        return
    job      = json.loads(job_file.read_text())
    log_file = UPLOAD_DIR / (uid + "_log.json")
    err_file = UPLOAD_DIR / (uid + "_error.txt")
    logs     = []

    def append(msg, color="#e2e8f0", pct=None, label=None):
        entry = {"msg": msg, "color": color}
        if pct   is not None: entry["pct"]   = pct
        if label is not None: entry["label"] = label
        logs.append(entry)
        try:
            log_file.write_text(json.dumps(logs))
        except Exception:
            pass

    def run():
        try:
            from coho_extractor import BankStatementParser, compute_analytics, fill_coho_template
            append("File: " + job["filename"] + " (" + str(job["size_mb"]) + " MB)", "#94a3b8", 5, "Opening PDF...")
            parser  = BankStatementParser(job["pdf_path"])
            n_pages = len(parser.pages)
            append("Pages found: " + str(n_pages), "#93c5fd", 15, "Reading pages...")
            header = parser._parse_header()
            append("Bank:    " + str(header.get("bank", "?")), "#e2e8f0", 25, "Parsing header...")
            append("Holder:  " + str(header.get("holder_name", "?")))
            append("Account: " + str(header.get("account_no", "?")))
            append("Parsing transactions...", "#93c5fd", 40, "Parsing transactions...")
            txns = parser._parse_transactions()
            append("Transactions found: " + str(len(txns)), "#4ade80", 58, "Computing analytics...")
            txns = parser._fix_balances(header.get("opening_balance", 0), txns)
            data = {"header": header, "transactions": txns,
                    "parsed_at": datetime.now().isoformat(), "source_file": job["pdf_path"]}
            an = compute_analytics(data)
            append("Total In:  " + _sym(header.get("currency","USD"))[0] + " " + ("{:,.0f}" if header.get("currency","USD").upper()=="KHR" else "{:,.2f}").format(an.get("total_in_amt", 0)) +
                   " (" + str(an.get("total_in_trx", 0)) + " trx)", "#4ade80", 72, "Generating Excel...")
            append("Total Out: " + _sym(header.get("currency","USD"))[0] + " " + ("{:,.0f}" if header.get("currency","USD").upper()=="KHR" else "{:,.2f}").format(an.get("total_out_amt", 0)) +
                   " (" + str(an.get("total_out_trx", 0)) + " trx)", "#f87171")
            xlsx = str(UPLOAD_DIR / (uid + "_COHO_Summary.xlsx"))
            fill_coho_template(data, xlsx)
            append("Excel generated!", "#4ade80", 88, "Saving result...")
            (UPLOAD_DIR / (uid + "_result.json")).write_text(json.dumps({
                "uid": uid,
                "header": {k: str(v) for k, v in header.items()},
                "analytics": {k: v for k, v in an.items() if k not in ("monthly_rows", "daily_closing")},
                "monthly_rows": [{k2: str(v2) if hasattr(v2, "strftime") else v2 for k2, v2 in mr.items()}
                                  for mr in an.get("monthly_rows", [])],
                "total_transactions": len(txns),
                "transactions": [{"date": t["date"].isoformat(), "desc": t["desc"],
                                   "money_in": t["money_in"], "money_out": t["money_out"],
                                   "os_balance": t["os_balance"],
                                   "is_closing": t["date"] in an.get("daily_closing", {}) and
                                   abs(t["os_balance"] - an.get("daily_closing", {}).get(t["date"], 99999)) < 0.01}
                                  for t in txns],
            }))
            append("Done!", "#4ade80", 100, "Complete!")
        except Exception as e:
            import traceback
            err_file.write_text(str(e))
            append("Error: " + str(e), "#f87171")
            append(traceback.format_exc(), "#f87171")

    threading.Thread(target=run, daemon=True).start()


@app.get("/poll/{uid}")
def poll(uid: str, from_: int = 0):
    log_file  = UPLOAD_DIR / (uid + "_log.json")
    done_file = UPLOAD_DIR / (uid + "_result.json")
    err_file  = UPLOAD_DIR / (uid + "_error.txt")
    logs = []; pct = 5; label = "Processing..."
    if log_file.exists():
        try:
            all_logs = json.loads(log_file.read_text())
            logs     = all_logs[from_:]
            if all_logs:
                pct   = all_logs[-1].get("pct", pct)
                label = all_logs[-1].get("label", label)
        except Exception:
            pass
    return JSONResponse({
        "logs": logs, "pct": pct, "label": label,
        "done": done_file.exists(),
        "error": err_file.read_text() if err_file.exists() else None,
    })


@app.get("/result/{uid}", response_class=HTMLResponse)
def result(uid: str):
    rf = UPLOAD_DIR / (uid + "_result.json")
    if not rf.exists():
        return HTMLResponse(page("Error",
            '<div style="color:red;padding:20px">Result not found. <a href="/coho/">Upload again</a></div>'))
    data   = json.loads(rf.read_text())
    header = data.get("header", {})
    an     = data.get("analytics", {})
    txns   = data.get("transactions", [])
    mrows  = data.get("monthly_rows", [])

    sym, fmt = _sym(header.get("currency","USD"))
    stats = (
        '<div class="stat-grid">'
        '<div class="stat"><div class="stat-num">' + str(data.get("total_transactions", 0)) + '</div>'
        '<div class="stat-lbl">Total Transactions</div></div>'
        '<div class="stat"><div class="stat-num" style="color:#059669">' + sym + ' ' +
        fmt(an.get("total_in_amt", 0)) + '</div>'
        '<div class="stat-lbl">Total In (' + str(an.get("total_in_trx", 0)) + ' trx)</div></div>'
        '<div class="stat"><div class="stat-num" style="color:#dc2626">' + sym + ' ' +
        fmt(an.get("total_out_amt", 0)) + '</div>'
        '<div class="stat-lbl">Total Out (' + str(an.get("total_out_trx", 0)) + ' trx)</div></div>'
        '<div class="stat"><div class="stat-num">' + sym + ' ' +
        fmt(an.get("avg_closing_bal", 0)) + '</div>'
        '<div class="stat-lbl">Avg Closing Balance</div></div>'
        '<div class="stat"><div class="stat-num">' + sym + ' ' +
        fmt(an.get("highest_balance", 0)) + '</div>'
        '<div class="stat-lbl">Highest Balance</div></div>'
        '<div class="stat"><div class="stat-num">' + sym + ' ' +
        fmt(an.get("lowest_balance", 0)) + '</div>'
        '<div class="stat-lbl">Lowest Balance</div></div>'
        '<div class="stat"><div class="stat-num">' + str(an.get("period_months", 0)) + '</div>'
        '<div class="stat-lbl">Months</div></div>'
        '<div class="stat"><div class="stat-num">' + sym + ' ' +
        fmt(an.get("reversal_amt", 0)) + '</div>'
        '<div class="stat-lbl">Reversals</div></div>'
        '</div>'
    )

    mhtml = '<table class="month-table"><thead><tr><th>#</th><th>Month</th><th>Debit Trx</th><th>Debit Amt</th><th>Credit Trx</th><th>Credit Amt</th><th>Avg Bal</th><th>Lowest</th><th>Highest</th></tr></thead><tbody>'
    for mr in mrows:
        mhtml += ('<tr><td>' + str(mr.get("no", "")) + '</td><td>' +
                  str(mr.get("month", ""))[:7] + '</td><td style="color:#dc2626">' +
                  str(mr.get("debit_trx", 0)) + '</td><td style="color:#dc2626">' +
                  fmt(mr.get("debit_amt", 0)) + '</td><td style="color:#059669">' +
                  str(mr.get("credit_trx", 0)) + '</td><td style="color:#059669">' +
                  fmt(mr.get("credit_amt", 0)) + '</td><td>' +
                  fmt(mr.get("avg_bal", 0)) + '</td><td>' +
                  fmt(mr.get("lowest_bal", 0)) + '</td><td>' +
                  fmt(mr.get("highest_bal", 0)) + '</td></tr>')
    mhtml += ('<tr class="total-row"><td colspan="2">Total</td><td style="color:#dc2626">' +
              str(an.get("total_out_trx", 0)) + '</td><td style="color:#dc2626">' +
              fmt(an.get("total_out_amt", 0)) +
              '</td><td style="color:#059669">' + str(an.get("total_in_trx", 0)) +
              '</td><td style="color:#059669">' +
              fmt(an.get("total_in_amt", 0)) +
              '</td><td colspan="3"></td></tr></tbody></table>')

    thtml = ""
    for i, t in enumerate(txns[:300]):
        out = float(t.get("money_out", 0))
        inn = float(t.get("money_in", 0))
        bal = float(t.get("os_balance", 0))
        cls = "closing" if t.get("is_closing") else ("debit" if out > 0 else "")
        thtml += ('<tr class="' + cls + '"><td style="color:#94a3b8;text-align:center">' +
                  str(i + 1) + '</td><td style="white-space:nowrap">' + t["date"][:10] +
                  '</td><td>' + t["desc"][:70] + '</td><td style="color:#dc2626;font-family:monospace">' +
                  (fmt(out) if out > 0 else "") +
                  '</td><td style="color:#059669;font-family:monospace">' +
                  (fmt(inn) if inn > 0 else "") +
                  '</td><td style="font-family:monospace">' + fmt(bal) +
                  '</td><td>' + ("&#127761;" if t.get("is_closing") else "") + '</td></tr>')

    period = str(header.get("period_from", ""))[:10] + " to " + str(header.get("period_to", ""))[:10]

    body = (
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">'
        '<div><h1 style="font-size:20px;font-weight:700">Extraction Complete</h1>'
        '<p style="font-size:13px;color:#64748b;margin-top:3px">' + period + '</p></div>'
        '<div style="display:flex;gap:10px">'
        '<a href="/coho/download/' + uid + '" class="btn btn-primary btn-lg">Download Excel</a>'
        '<a href="/coho/" class="btn btn-ghost">Upload Another</a></div></div>'
        '<div class="card"><div class="card-title">Account Information</div>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:12px">'
        '<div><span style="color:#64748b">Bank: </span><strong>' + str(header.get("bank", "")) + '</strong></div>'
        '<div><span style="color:#64748b">Holder: </span><strong>' + str(header.get("holder_name", "")) + '</strong></div>'
        '<div><span style="color:#64748b">Account: </span><strong>' + str(header.get("account_no", "")) + '</strong></div>'
        '<div><span style="color:#64748b">Currency: </span><strong>' + str(header.get("currency", "")) + '</strong></div>'
        '</div></div>'
        '<div class="card"><div class="card-title">COHO Summary</div>' + stats + '</div>'
        '<div class="card"><div class="card-title">Monthly Breakdown</div>'
        '<div style="overflow-x:auto">' + mhtml + '</div></div>'
        '<div class="card"><div class="card-title">Transactions</div>'
        '<div style="overflow-x:auto"><table class="trx-table">'
        '<thead><tr><th>#</th><th>Date</th><th>Description</th>'
        '<th>Debit</th><th>Credit</th><th>Balance</th><th></th></tr></thead>'
        '<tbody>' + thtml + '</tbody></table></div></div>'
        '<div style="text-align:center;padding:20px">'
        '<a href="/coho/download/' + uid + '" class="btn btn-primary btn-lg">Download COHO Excel</a></div>'
    )
    return HTMLResponse(page("Results", body))


@app.get("/download/{uid}")
def download(uid: str):
    matches = list(UPLOAD_DIR.glob(uid + "_COHO_Summary.xlsx"))
    if not matches:
        return JSONResponse({"error": "not found"}, status_code=404)
    raw   = matches[0].read_bytes()
    fname = "COHO_" + datetime.now().strftime("%Y%m%d_%H%M") + ".xlsx"
    return StreamingResponse(io.BytesIO(raw),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="' + fname + '"'})


@app.get("/health")
def health():
    return {"status": "ok"}
