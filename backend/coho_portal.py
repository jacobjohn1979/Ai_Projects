"""
coho_portal.py — Credit Assessment Tools — FI Statement Analyser
Upload PDF → extract transactions → download FI Statement Excel
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

app = FastAPI(title="FI Statement Analyser")

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
@media print {
  body { background: white !important; font-size: 11px !important; }
  .topbar { display: none !important; }
  .no-print { display: none !important; }
  .card { box-shadow: none !important; border: 1px solid #ccc !important; 
          margin-bottom: 10px !important; break-inside: avoid; }
  .stat-grid { grid-template-columns: repeat(4,1fr) !important; }
  .btn { display: none !important; }
  a.btn { display: none !important; }
  .print-header { display: block !important; }
  @page { margin: 1.5cm; size: A4; }
}
.print-header { display: none; }

.month-table th{background:var(--blue-light);color:var(--blue-dark);padding:8px;
                text-align:center;font-weight:700;border:1px solid #bdd7ee}
.month-table td{padding:7px 8px;border:1px solid #e2e8f0;text-align:right}
.month-table td:nth-child(1),.month-table td:nth-child(2){text-align:center}
.month-table tr:nth-child(even) td{background:#f8fafc}
.month-table .total-row td{background:#fffde7;font-weight:700}
"""
UPLOAD_JS = """
<script>
function selectFile(input) {
  var file = input.files[0];
  if (!file) return;
  document.getElementById('fname').textContent = file.name;
  document.getElementById('fsize').textContent = '(' + (file.size/1024/1024).toFixed(2) + ' MB)';
  document.getElementById('info').style.display = 'block';
}
function uploadFile() {
  var input = document.getElementById('inp');
  if (!input.files[0]) return;
  var form = new FormData();
  form.append('pdf_file', input.files[0]);
  fetch('/coho/extract', {method:'POST', body:form})
    .then(function(r){ return r.json(); })
    .then(function(d){ if(d.redirect) window.location.href = d.redirect; })
    .catch(function(e){ alert('Upload failed: ' + e); });
}
var zone = document.getElementById('zone');
if (zone) {
  zone.addEventListener('dragover', function(e){ e.preventDefault(); zone.style.borderColor='#2563eb'; });
  zone.addEventListener('dragleave', function(){ zone.style.borderColor=''; });
  zone.addEventListener('drop', function(e){
    e.preventDefault(); zone.style.borderColor='';
    var f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.pdf')) {
      var dt = new DataTransfer(); dt.items.add(f);
      var inp = document.getElementById('inp');
      inp.files = dt.files; selectFile(inp);
    }
  });
}
</script>
"""


def _sym(currency):
    """Return currency symbol and formatter based on currency code."""
    if str(currency).upper() == "KHR":
        return "KHR", lambda v: "{:,.0f}".format(float(v))  # no decimals for KHR
    return "$", lambda v: "{:,.2f}".format(float(v))


def page(title, body, request=None):
    role = "viewer"
    if request:
        try:
            import jwt as _jwt, os as _os
            token = request.cookies.get("auth_token","")
            if token:
                payload = _jwt.decode(token, _os.getenv("JWT_SECRET",""), algorithms=["HS256"])
                role = payload.get("role","viewer")
        except: pass

    # Build nav based on role
    nav_items = [('📄 FI Statement', '/coho/')]
    if role in ('admin','credit_officer','cbc_manager','viewer'):
        nav_items.append(('📊 CBC', '/cbc/'))
    nav_items.append(('📋 My Jobs', '/coho/jobs'))
    if role == 'admin':
        nav_items.append(('⚙ Admin', '/auth/admin'))

    nav_html = '\n    '.join(
        f'<a href="{url}" class="nav-link">{label}</a>'
        for label, url in nav_items
    )
    role_pill = (f'<span style="font-size:10px;padding:2px 8px;border-radius:10px;'
                 f'background:rgba(255,255,255,.15);color:#cbd5e1;margin-right:8px">'
                 f'{role.replace("_"," ").title()}</span>') if role != "viewer" else ""
    logout = '<a href="/auth/logout" class="nav-link" style="color:#f87171;border:1px solid rgba(248,113,113,.3);border-radius:6px;padding:5px 12px">→ Logout</a>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Credit Assessment Tools - FI Statement Analyser - {title}</title>
<style>{CSS}</style></head>
<body>
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">FIS</div>
    <div><div class="brand-title">Credit Assessment Tools</div>
         <div class="brand-sub">FI Statement Analyser</div></div>
  </div>
  <div class="nav-links">
    {role_pill}
    {nav_html}
    {logout}
  </div>
</div>
<div class="content">{body}</div>
</body></html>""" + UPLOAD_JS


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    role = _get_role(request)

    trainer_link = ""
    if role == "admin":
        trainer_link = "<a href='/trainer/' style='color:#2563eb;text-decoration:none;font-size:11px'>Open Trainer &rarr;</a>"

    extracts = [
        ("📅","Statement Period","From/To dates"),
        ("👤","Account Holder","Name & account number"),
        ("💱","Currency","USD or KHR auto-detected"),
        ("📥","Money In","Total credits & count"),
        ("📤","Money Out","Total debits & count"),
        ("💰","Balance","Running balance per row"),
        ("📊","Monthly Summary","In/out/avg per month"),
        ("🔁","Reversals","Flagged reversed transactions"),
        ("📋","All Transactions","Date, desc, amount, balance"),
    ]
    extract_html = ""
    for ic, nm, desc in extracts:
        extract_html += (
            '<div style="display:flex;align-items:flex-start;gap:8px;padding:8px;'
            'background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0">'
            '<span style="font-size:18px;min-width:24px">' + ic + '</span>'
            '<div><div style="font-size:12px;font-weight:700;color:#1F4E79">' + nm + '</div>'
            '<div style="font-size:11px;color:#64748b">' + desc + '</div></div></div>'
        )

    banks = [
        ("ABA Bank","🏦","USD/KHR"), ("Wing Bank","🦅","USD/KHR"),
        ("ACLEDA Bank","🏛","KHR"),  ("KB Prasac","💳","USD/KHR"),
        ("Philip Bank","🏦","USD"),  ("AMRET MFI","🌾","USD"),
        ("Woori Bank","🏢","USD"),   ("Hattha Bank","💰","USD/KHR"),
        ("Canadia Bank","🏦","KHR"), ("Sathapana Bank","🏛","USD"),
        ("Post Bank","📮","USD"),    ("Maybank","🏦","USD"),
        ("AMK Microfinance","🌱","USD"), ("Taiwan Coop Bank","🏦","USD"),
        ("LOLC Cambodia","🏦","USD"),
    ]
    bank_html = ""
    for nm, ic, cy in banks:
        bank_html += (
            '<div style="display:flex;align-items:center;gap:6px;padding:6px 12px;'
            'background:#EAF2F8;border-radius:20px;font-size:11px;font-weight:600;color:#1F4E79">'
            + ic + ' ' + nm +
            '<span style="color:#94a3b8;font-weight:400;margin-left:2px">' + cy + '</span>'
            '</div>'
        )

    body = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:18px">'

        # Upload card
        '<div class="card">'
        '<div class="card-title">Upload FI Statement PDF</div>'
        '<div class="upload-zone" id="zone" onclick="document.getElementById(\'inp\').click()">'
        '<div style="font-size:32px;margin-bottom:8px">&#128196;</div>'
        '<div style="font-size:15px;font-weight:600;margin-bottom:6px">Drop PDF here or click to browse</div>'
        '<div style="font-size:12px;color:#64748b">15 supported banks &mdash; Max 30MB</div>'
        '<input type="file" id="inp" accept=".pdf" style="display:none" onchange="selectFile(this)">'
        '</div>'
        '<div id="info" style="display:none;margin-top:12px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:10px;background:#f8fafc;border-radius:8px;margin-bottom:10px">'
        '<span id="fname" style="font-weight:600;font-size:13px"></span>'
        '<span id="fsize" style="color:#64748b;font-size:12px"></span>'
        '</div>'
        '<button class="btn btn-primary" style="width:100%;padding:12px;font-size:14px" '
        'onclick="uploadFile()">&#x1F4E4; Extract Transactions</button>'
        '</div>'
        '</div>'

        # What gets extracted
        '<div class="card">'
        '<div class="card-title">What Gets Extracted</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
        + extract_html +
        '</div></div>'

        '</div>'

        # Supported banks
        '<div class="card">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
        '<div class="card-title" style="margin:0">Supported Banks &amp; MFIs</div>'
        '<span style="font-size:11px;color:#94a3b8">15 institutions &mdash; auto-detected, no setup needed</span>'
        '</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:8px">' + bank_html + '</div>'
        '<div style="margin-top:10px;font-size:11px;color:#94a3b8;display:flex;gap:12px;align-items:center">'
        '<span>&#10003; KHR &amp; USD supported</span>'
        '<span>&#10003; Multi-page PDFs</span>'
        '<span>&#10003; Encrypted PDFs auto-handled</span>'
        + trainer_link +
        '</div>'
        '</div>'
    )
    return HTMLResponse(page("Upload", body, request))



def _get_username(request: Request) -> str:
    """Get username from JWT cookie."""
    try:
        import jwt as _jwt
        import os as _os
        token = request.cookies.get("auth_token","")
        if not token:
            return "unknown"
        secret = _os.getenv("JWT_SECRET","")
        payload = _jwt.decode(token, secret, algorithms=["HS256"])
        return payload.get("sub","unknown")
    except:
        return "unknown"

def _get_role(request: Request) -> str:
    """Get role from JWT cookie."""
    try:
        import jwt as _jwt
        import os as _os
        token = request.cookies.get("auth_token","")
        if not token:
            return "staff"
        secret = _os.getenv("JWT_SECRET","")
        payload = _jwt.decode(token, secret, algorithms=["HS256"])
        return payload.get("role","staff")
    except:
        return "staff"

@app.post("/extract")
async def extract(request: Request, pdf_file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uid      = str(uuid.uuid4())[:8]
    pdf_path = UPLOAD_DIR / (uid + "_" + pdf_file.filename)
    raw      = await pdf_file.read()
    pdf_path.write_bytes(raw)
    (UPLOAD_DIR / (uid + "_job.json")).write_text(json.dumps({
        "uid": uid, "filename": pdf_file.filename,
        "pdf_path": str(pdf_path),
        "size_mb": round(len(raw) / 1024 / 1024, 2),
        "username": _get_username(request),
    }))
    return JSONResponse({"uid": uid})


@app.get("/progress/{uid}", response_class=HTMLResponse)
def progress(uid: str, request: Request):
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
    return HTMLResponse(page("Processing", body, request))


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
def result(uid: str, request: Request):
    rf = UPLOAD_DIR / (uid + "_result.json")
    if not rf.exists():
        return HTMLResponse(page("Error",
            request, '<div style="color:red;padding:20px">Result not found. <a href="/coho/">Upload again</a></div>'))
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
        '<a href="/coho/download/' + uid + '" class="btn btn-primary btn-lg">&#x2B07; Download Excel</a>'
        '<button onclick="window.print()" class="btn btn-ghost no-print" style="border-color:#1F4E79;color:#1F4E79">&#x1F5B6; Print / Save PDF</button>'
        '<a href="/coho/" class="btn btn-ghost no-print">Upload Another</a></div></div>'
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
        '<a href="/coho/download/' + uid + '" class="btn btn-primary btn-lg no-print">&#x2B07; Download Excel</a>'
        '<button onclick="window.print()" class="btn btn-ghost" style="border-color:#1F4E79;color:#1F4E79;margin-left:10px">&#x1F5B6; Print / Save PDF</button></div>'
    )
    return HTMLResponse(page("Results", body, request))


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



@app.get("/debug", response_class=HTMLResponse)
def debug_jobs(request: Request):
    """Show all jobs with their status - for debugging failed extractions."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    current_user = _get_username(request)
    current_role = _get_role(request)
    jobs = sorted(UPLOAD_DIR.glob("*_job.json"),
                  key=lambda f: f.stat().st_mtime, reverse=True)
    # Filter by user unless admin
    if current_role != "admin":
        filtered = []
        for jf in jobs:
            try:
                j = json.loads(jf.read_text())
                if j.get("username","unknown") == current_user:
                    filtered.append(jf)
            except: pass
        jobs = filtered

    rows = ""
    for jf in jobs[:50]:
        uid = jf.stem.replace("_job","")
        try:
            job  = json.loads(jf.read_text())
            done = (UPLOAD_DIR / (uid+"_result.json")).exists()
            err  = (UPLOAD_DIR / (uid+"_error.txt"))
            log  = (UPLOAD_DIR / (uid+"_log.json"))

            error_text = err.read_text()[:300] if err.exists() else ""
            log_msgs   = ""
            if log.exists():
                try:
                    logs = json.loads(log.read_text())
                    log_msgs = " | ".join(l["msg"] for l in logs[-3:] if l.get("msg"))[:200]
                except: pass

            status = "done" if done else ("error" if err.exists() else "processing")
            color  = "#059669" if done else ("#dc2626" if err.exists() else "#d97706")

            rows += f"""<tr>
              <td style="white-space:nowrap;font-size:11px">{uid}</td>
              <td style="font-size:12px">{job.get("filename","?")}</td>
              <td style="font-size:11px">{job.get("size_mb",0)} MB</td>
              <td><span style="color:{color};font-weight:600">{status}</span></td>
              <td style="font-size:11px;color:#64748b">{error_text or log_msgs}</td>
              <td>
                {"<a href='/coho/result/"+uid+"' style='font-size:11px'>View</a>" if done else ""}
                {"<a href='/coho/retry/"+uid+"' style='font-size:11px;color:#dc2626'>Retry</a>" if err.exists() else ""}
              </td>
            </tr>"""
        except Exception as e:
            rows += f"<tr><td colspan='6' style='color:red'>{uid}: {e}</td></tr>"

    body = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <h1 style="font-size:20px;font-weight:700">Job Debug Panel</h1>
      <a href="/coho/" class="btn btn-ghost">Back</a>
    </div>
    <div class="card">
      <div class="card-title">Recent Jobs ({len(jobs)} total)</div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr style="background:#D6E4F0">
            <th style="padding:8px;text-align:left">UID</th>
            <th style="padding:8px;text-align:left">File</th>
            <th style="padding:8px">Size</th>
            <th style="padding:8px">Status</th>
            <th style="padding:8px;text-align:left">Details</th>
            <th style="padding:8px">Action</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""
    return HTMLResponse(page("Debug", body, request))


@app.get("/retry/{uid}", response_class=HTMLResponse)
def retry_job(uid: str, request: Request):
    """Clear error and retry extraction."""
    err_file = UPLOAD_DIR / (uid + "_error.txt")
    log_file = UPLOAD_DIR / (uid + "_log.json")
    if err_file.exists(): err_file.unlink()
    if log_file.exists(): log_file.unlink()
    _start_extraction(uid)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(f"/coho/progress/{uid}", status_code=303)


@app.get("/pdf/{uid}")
def pdf_summary(uid: str):
    """Generate a PDF summary of the COHO result."""
    rf = UPLOAD_DIR / (uid + "_result.json")
    if not rf.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    data   = json.loads(rf.read_text())
    header = data.get("header", {})
    an     = data.get("analytics", {})
    mrows  = data.get("monthly_rows", [])

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               topMargin=1.5*cm, bottomMargin=1.5*cm,
                               leftMargin=1.5*cm, rightMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story  = []

    navy   = colors.HexColor("#1F4E79")
    light  = colors.HexColor("#D6E4F0")
    white  = colors.white
    red    = colors.HexColor("#dc2626")
    green  = colors.HexColor("#059669")

    title_style = ParagraphStyle("title", fontSize=16, fontName="Helvetica-Bold",
                                  textColor=navy, alignment=TA_CENTER, spaceAfter=6)
    sub_style   = ParagraphStyle("sub", fontSize=10, textColor=colors.grey,
                                  alignment=TA_CENTER, spaceAfter=12)
    label_style = ParagraphStyle("label", fontSize=9, fontName="Helvetica-Bold",
                                  textColor=navy)

    currency = header.get("currency","USD")
    sym, fmt = ("KHR", lambda v: "{:,.0f}".format(float(v))) if currency=="KHR"                else ("$", lambda v: "{:,.2f}".format(float(v)))

    period = str(header.get("period_from",""))[:10] + " to " + str(header.get("period_to",""))[:10]

    story.append(Paragraph("CREDIT ASSESSMENT TOOLS — FI STATEMENT SUMMARY", title_style))
    story.append(Paragraph(f"Period: {period}", sub_style))
    story.append(Spacer(1, 0.3*cm))

    # Account info table
    info_data = [
        ["Bank", str(header.get("bank","")),      "Account No.", str(header.get("account_no",""))],
        ["Holder", str(header.get("holder_name","")), "Currency", str(header.get("currency",""))],
        ["Opening Bal.", sym+" "+fmt(header.get("opening_balance",0)),
         "Closing Bal.", sym+" "+fmt(header.get("ending_balance",0))],
    ]
    info_table = Table(info_data, colWidths=[3*cm,6*cm,3*cm,6*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), light),
        ("FONTNAME",   (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",   (2,0),(2,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 9),
        ("GRID",       (0,0),(-1,-1), 0.5, colors.white),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[light, colors.HexColor("#EAF2F8")]),
        ("TOPPADDING", (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # Summary stats
    stats_data = [
        ["Total Transactions", "Total In (Trx)", "Total Out (Trx)", "Avg Balance", "Months"],
        [str(data.get("total_transactions",0)),
         sym+" "+fmt(an.get("total_in_amt",0))+" ("+str(an.get("total_in_trx",0))+")",
         sym+" "+fmt(an.get("total_out_amt",0))+" ("+str(an.get("total_out_trx",0))+")",
         sym+" "+fmt(an.get("avg_closing_bal",0)),
         str(an.get("period_months",0))],
        ["Highest Balance", "Lowest Balance", "Avg Monthly In", "Avg Monthly Out", "Reversals"],
        [sym+" "+fmt(an.get("highest_balance",0)),
         sym+" "+fmt(an.get("lowest_balance",0)),
         sym+" "+fmt(an.get("avg_monthly_in_amt",0)),
         sym+" "+fmt(an.get("avg_monthly_out_amt",0)),
         sym+" "+fmt(an.get("reversal_amt",0))],
    ]
    stats_table = Table(stats_data, colWidths=[3.6*cm]*5)
    stats_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), navy),
        ("TEXTCOLOR",   (0,0),(-1,0), white),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.white),
        ("BACKGROUND",  (0,1),(-1,1), light),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.4*cm))

    # Monthly breakdown
    story.append(Paragraph("Monthly Breakdown", ParagraphStyle("h2", fontSize=11,
                fontName="Helvetica-Bold", textColor=navy, spaceAfter=6)))

    m_header = ["#", "Month", "Debit\nTrx", "Debit Amt", "Credit\nTrx", "Credit Amt",
                "Avg Bal", "Lowest Bal", "Highest Bal"]
    m_data   = [m_header]
    for mr in mrows:
        m_data.append([
            str(mr.get("no","")),
            str(mr.get("month",""))[:7],
            str(mr.get("debit_trx",0)),
            fmt(mr.get("debit_amt",0)),
            str(mr.get("credit_trx",0)),
            fmt(mr.get("credit_amt",0)),
            fmt(mr.get("avg_bal",0)),
            fmt(mr.get("lowest_bal",0)),
            fmt(mr.get("highest_bal",0)),
        ])
    # Totals row
    m_data.append([
        "","TOTAL",
        str(an.get("total_out_trx",0)), fmt(an.get("total_out_amt",0)),
        str(an.get("total_in_trx",0)),  fmt(an.get("total_in_amt",0)),
        "","",""
    ])

    col_w = [0.8*cm, 2*cm, 1.6*cm, 2.2*cm, 1.6*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm]
    m_table = Table(m_data, colWidths=col_w)
    m_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), navy),
        ("TEXTCOLOR",     (0,0),(-1,0), white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ALIGN",         (2,0),(-1,-1), "RIGHT"),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.lightgrey),
        ("ROWBACKGROUNDS",(0,1),(-2,-1),[white, light]),
        ("BACKGROUND",    (0,-1),(-1,-1), colors.HexColor("#FFF9C4")),
        ("FONTNAME",      (0,-1),(-1,-1), "Helvetica-Bold"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 0.4*cm))

    # Footer
    from datetime import datetime
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Credit Assessment Tools",
        ParagraphStyle("footer", fontSize=7, textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    fname = f"COHO_Summary_{uid}.pdf"
    return StreamingResponse(buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{fname}"'})



@app.get("/jobs", response_class=HTMLResponse)
def jobs_history(request: Request):
    """Unified jobs history — shows COHO + CBC + future modules."""
    current_user = _get_username(request)
    current_role = _get_role(request)
    import datetime

    # Define all modules and their upload dirs
    MODULE_DIRS = {
        "COHO": (UPLOAD_DIR, "/coho/result/", "/coho/download/", "/coho/retry/"),
        "CBC":  (UPLOAD_DIR.parent / "cbc", "/cbc/result/", "/cbc/download/", None),
    }

    all_jobs = []
    for module, (d, view_url, dl_url, retry_url) in MODULE_DIRS.items():
        if not d.exists():
            continue
        for jf in d.glob("*_job.json"):
            try:
                job = json.loads(jf.read_text())
                job["_module"]    = module
                job["_jf"]        = jf
                job["_view_url"]  = view_url
                job["_dl_url"]    = dl_url
                job["_retry_url"] = retry_url
                job["_mtime"]     = jf.stat().st_mtime
                # Admin sees all; others see only their own jobs
                job_user = job.get("username", "unknown")
                if current_role != "admin" and job_user != current_user:
                    continue
                all_jobs.append(job)
            except: pass

    all_jobs.sort(key=lambda j: j["_mtime"], reverse=True)

    counts = {"done":0, "error":0, "processing":0}
    _clear = ('<button onclick="fetch(\'/coho/jobs/clear\',{method:\'POST\'}).then(()=>location.reload())"'
              ' class="btn btn-ghost" style="color:#dc2626">&#x1F5D1; Clear Failed</button>'
              ) if current_role == "admin" else ""
    _clear = ""
    if current_role == "admin":
        _clear = ('<button onclick="fetch(\'/coho/jobs/clear\',{method:\'POST\'}).then(()=>location.reload())"'
                  ' class="btn btn-ghost" style="color:#dc2626">&#x1F5D1; Clear Failed</button>')
    rows = ""

    for job in all_jobs[:200]:
        uid      = job["uid"]
        module   = job["_module"]
        jf       = job["_jf"]
        d        = jf.parent
        view_url = job["_view_url"]
        dl_url   = job["_dl_url"]
        retry_url= job["_retry_url"]

        done     = (d / (uid+"_result.json")).exists()
        err_file = d / (uid+"_error.txt")

        if done:
            status = "done"; counts["done"] += 1
            badge  = '<span style="color:#059669;font-weight:600">&#x2713; Done</span>'
        elif err_file.exists():
            status = "error"; counts["error"] += 1
            badge  = '<span style="color:#dc2626;font-weight:600">&#x2717; Error</span>'
        else:
            # Mark as stale if older than 2 hours with no result
            import time
            age_hours = (time.time() - job["_mtime"]) / 3600
            if age_hours > 2:
                status = "error"; counts["error"] += 1
                badge  = '<span style="color:#94a3b8;font-weight:600">&#x2715; Stale</span>'
            else:
                status = "processing"; counts["processing"] += 1
                badge  = '<span style="color:#d97706;font-weight:600">&#x23F3; Processing</span>'

        # Summary
        summary = ""
        if done:
            try:
                r = json.loads((d/(uid+"_result.json")).read_text())
                if module == "COHO":
                    an  = r.get("analytics",{})
                    h   = r.get("header",{})
                    cur = h.get("currency","USD")
                    sym = "KHR " if cur=="KHR" else "$ "
                    fmt = (lambda v: f"{float(v):,.0f}") if cur=="KHR" else (lambda v: f"{float(v):,.2f}")
                    summary = (f'<span style="color:#059669">&#8593;{sym}{fmt(an.get("total_in_amt",0))}</span> '
                               f'<span style="color:#dc2626">&#8595;{sym}{fmt(an.get("total_out_amt",0))}</span> '
                               f'| {r.get("total_transactions",0)} txns')
                else:
                    apps = r.get("applicants",[])
                    summary = f"{len(apps)} applicant(s)"
                    if apps:
                        p = apps[0].get("personal",{})
                        summary += f" | {p.get('full_name_en','')}"
            except: pass
        elif err_file.exists():
            summary = f'<span style="color:#dc2626;font-size:11px">{err_file.read_text()[:60]}</span>'

        mtime    = datetime.datetime.fromtimestamp(job["_mtime"]).strftime("%Y-%m-%d %H:%M")
        user_col = f'<td style="font-size:11px;color:#64748b">{job.get("username","?")}</td>' if current_role=="admin" else ""
        mod_col  = f'<td><span style="font-size:10px;padding:2px 8px;border-radius:10px;background:#dbeafe;color:#1e40af">{module}</span></td>'

        actions  = ""
        if done:
            actions += f'<a href="{view_url}{uid}" style="font-size:12px;color:#2563eb;margin-right:8px">View</a>'
            actions += f'<a href="{dl_url}{uid}" style="font-size:12px;color:#059669;margin-right:8px">Excel</a>'
        if status == "error" and retry_url:
            actions += f'<a href="{retry_url}{uid}" style="font-size:12px;color:#dc2626">Retry</a>'

        rows += f"""<tr>
          <td style="font-size:11px;color:#94a3b8;white-space:nowrap">{mtime}</td>
          {mod_col}
          {user_col}
          <td style="font-size:12px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
              title="{job.get('filename','?')}">{job.get('filename','?')}</td>
          <td style="font-size:11px">{job.get('size_mb',0)} MB</td>
          <td>{badge}</td>
          <td style="font-size:11px">{summary}</td>
          <td style="white-space:nowrap">{actions}</td>
        </tr>"""

    user_hdr = "<th style='padding:8px;color:#1F4E79'>User</th>" if current_role=="admin" else ""
    body = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <div>
        <h1 style="font-size:20px;font-weight:700">My Jobs</h1>
        <p style="font-size:13px;color:#64748b;margin-top:3px">
          All credit assessment jobs &nbsp;|&nbsp;
          <span style="color:#059669">&#x2713; {counts['done']} done</span> &nbsp;|&nbsp;
          <span style="color:#dc2626">&#x2717; {counts['error']} failed</span> &nbsp;|&nbsp;
          <span style="color:#d97706">&#x23F3; {counts['processing']} running</span>
        </p>
      </div>
      <div style="display:flex;gap:10px">
        <a href="/coho/" class="btn btn-primary">+ COHO Upload</a>
        <a href="/cbc/"  class="btn btn-ghost">+ CBC Upload</a>
        {_clear}
        {_clear}
      </div>
    </div>
    <div class="card">
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr style="background:#D6E4F0">
            <th style="padding:8px;text-align:left;color:#1F4E79">Time</th>
            <th style="padding:8px;color:#1F4E79">Module</th>
            {user_hdr}
            <th style="padding:8px;text-align:left;color:#1F4E79">File</th>
            <th style="padding:8px;color:#1F4E79">Size</th>
            <th style="padding:8px;color:#1F4E79">Status</th>
            <th style="padding:8px;text-align:left;color:#1F4E79">Summary</th>
            <th style="padding:8px;color:#1F4E79">Actions</th>
          </tr></thead>
          <tbody>
            {rows if rows else
             "<tr><td colspan='8' style='text-align:center;padding:30px;color:#94a3b8'>No jobs yet</td></tr>"}
          </tbody>
        </table>
      </div>
    </div>"""
    return HTMLResponse(page("My Jobs", body, request))


@app.post("/jobs/clear")
def clear_old_jobs():
    """Remove failed and old job files."""
    removed = 0
    for f in UPLOAD_DIR.glob("*_error.txt"):
        uid = f.stem.replace("_error","")
        for ext in ["_error.txt","_log.json","_job.json"]:
            p = UPLOAD_DIR / (uid+ext)
            if p.exists(): p.unlink(); removed += 1
    return JSONResponse({"removed": removed})


@app.get("/health")
def health():
    return {"status": "ok"}
