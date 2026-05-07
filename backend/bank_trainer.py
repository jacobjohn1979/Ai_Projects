"""
bank_trainer.py — Bank Statement Format Trainer
Upload any bank PDF → auto-detect format → confirm → save bank profile
Profiles are used by coho_extractor.py for automatic parsing.

Port: 8009  Access: /trainer/
"""
import os, io, json, uuid, re, logging
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import pdfplumber
from dotenv import load_dotenv

load_dotenv()
log        = logging.getLogger("bank_trainer")
BASE_DIR   = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
TRAIN_DIR  = BASE_DIR / "trainer"
PROFILE_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads")) / "bank_profiles"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Bank Format Trainer")

CSS = """
:root{--nav:#1a2744;--accent:#2563eb;--ok:#059669;--ok-bg:#ecfdf5;
  --danger:#dc2626;--warn:#d97706;--warn-bg:#fffbeb;
  --surface:#f8fafc;--card:#fff;--border:#e2e8f0;
  --text:#0f172a;--muted:#64748b;--blue-dark:#1F4E79;--blue-light:#D6E4F0;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--surface);color:var(--text);font-size:14px}
.topbar{background:var(--nav);color:#fff;height:56px;display:flex;
        align-items:center;padding:0 24px;justify-content:space-between;
        box-shadow:0 2px 8px rgba(0,0,0,.2)}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;background:#dc2626;border-radius:7px;
            display:flex;align-items:center;justify-content:center;
            font-size:10px;font-weight:800;color:#fff}
.brand-title{font-size:14px;font-weight:700;color:#f1f5f9}
.brand-sub{font-size:11px;color:#94a3b8}
.nav-link{color:#94a3b8;font-size:12px;padding:6px 12px;border-radius:6px;text-decoration:none}
.nav-link:hover{background:rgba(255,255,255,.1);color:#fff}
.nav-links{display:flex;gap:6px}
.content{max-width:1200px;margin:0 auto;padding:24px 20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:20px 24px;margin-bottom:18px}
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
.btn-success{background:var(--ok);color:#fff}
.btn-danger{background:var(--danger);color:#fff}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-lg{padding:13px 28px;font-size:14px;font-weight:600}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge-ok{background:var(--ok-bg);color:var(--ok)}
.badge-warn{background:var(--warn-bg);color:var(--warn)}
.badge-blue{background:#eff6ff;color:var(--accent)}
.raw-text{background:#0f172a;color:#e2e8f0;font-family:monospace;font-size:11px;
          padding:16px;border-radius:8px;overflow-x:auto;white-space:pre;
          max-height:400px;overflow-y:auto;line-height:1.6}
.line-num{color:#475569;user-select:none;margin-right:12px}
.line-date{color:#4ade80}
.line-amt{color:#fbbf24}
.line-skip{color:#374151}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:var(--blue-light);color:var(--blue-dark);padding:8px;
   text-align:left;font-weight:700;border-bottom:2px solid #bdd7ee}
td{padding:7px 8px;border-bottom:1px solid #f1f5f9}
tr:hover td{background:#f8fafc}
.form-group{margin-bottom:14px}
label{display:block;font-size:12px;font-weight:600;color:var(--muted);margin-bottom:5px}
input,select,textarea{width:100%;padding:8px 12px;border:1px solid var(--border);
  border-radius:6px;font-size:13px;color:var(--text);background:#fff}
input:focus,select:focus{outline:none;border-color:var(--accent)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.profile-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
              padding:14px 16px;display:flex;justify-content:space-between;align-items:center}
.alert{padding:12px 16px;border-radius:8px;font-size:13px;margin-bottom:14px}
.alert-ok{background:var(--ok-bg);border:1px solid #6ee7b7;color:#065f46}
.alert-warn{background:var(--warn-bg);border:1px solid #fcd34d;color:#92400e}
.alert-err{background:#fef2f2;border:1px solid #fca5a5;color:#991b1b}
"""

def page(title, body):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bank Trainer - {title}</title>
<style>{CSS}</style></head>
<body>
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">TRAIN</div>
    <div><div class="brand-title">Bank Format Trainer</div>
         <div class="brand-sub">Add support for new bank statement formats</div></div>
  </div>
  <div class="nav-links">
    <a href="/trainer/" class="nav-link">Upload</a>
    <a href="/trainer/profiles" class="nav-link">Bank Profiles</a>
    <a href="/coho/" class="nav-link">COHO Portal</a>
  </div>
</div>
<div class="content">{body}</div>
</body></html>"""


# ── DETECT PATTERNS ───────────────────────────────────────────────────────────

DATE_PATTERNS = [
    ("DD-Mon-YYYY", r"\b(\d{2}-[A-Za-z]{3}-\d{4})\b",     "22-Apr-2026"),
    ("Mon DD, YYYY", r"\b([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\b", "Aug 01, 2025"),
    ("DD MON YY",   r"\b(\d{1,2}\s+[A-Z]{3}\s+\d{2})\b",  "24 MAR 23"),
    ("DD/MM/YYYY",  r"\b(\d{2}/\d{2}/\d{4})\b",            "01/02/2025"),
    ("DD/MM/YY",    r"\b(\d{2}/\d{2}/\d{2})\b",            "01/02/23"),
    ("YYYY-MM-DD",  r"\b(\d{4}-\d{2}-\d{2})\b",            "2025-08-01"),
]

SWIFT_PATTERNS = {
    "ABAAKHPP":   "ABA Bank",
    "WIGCKHPPXXX":"Wing Bank",
    "ACLBKHPP":   "ACLEDA Bank",
    "PPCBKHPP":   "Prince Bank",
    "WIGCKHPP":   "Wing Bank",
    "VRBBKHPP":   "Vattanac Bank",
    "PRASKHPP":   "PRASAC MFI",
    "WORICAMM":   "Woori Bank",
    "MBBECAMM":   "Maybank",
    "CTIBKHPP":   "Canadia Bank",
    "AMRJKHPP":   "Amret MFI",
    "ARDBKHPP":   "ARDB Bank",
    "CMBCKHPP":   "Chip Mong Bank",
    "FTBCKHPP":   "FTB Bank",
}

def auto_detect(text: str, pages: list) -> dict:
    """Auto-detect bank format from PDF text."""
    result = {
        "swift": "",
        "bank_name": "Unknown Bank",
        "date_pattern": "",
        "date_example": "",
        "currency": "USD",
        "has_khr": False,
        "has_usd": False,
        "keywords": [],
        "sample_lines": [],
        "all_lines": [],
    }

    # Detect SWIFT
    for swift, name in SWIFT_PATTERNS.items():
        if swift.upper() in text.upper():
            result["swift"]     = swift
            result["bank_name"] = name
            break

    # Detect currency
    result["has_usd"] = "USD" in text.upper()
    result["has_khr"] = "KHR" in text.upper()
    result["currency"] = "KHR" if (result["has_khr"] and not result["has_usd"]) else "USD"

    # Detect date pattern
    for name, pattern, example in DATE_PATTERNS:
        if re.search(pattern, text):
            result["date_pattern"] = name
            result["date_example"] = example
            break

    # Detect amount format
    if re.search(r"\d{1,3},\d{3},\d{3}\.", text):
        result["amount_format"] = "KHR (1,000,000.00)"
    elif re.search(r"\d+,\d{3}\.\d{2}", text):
        result["amount_format"] = "USD (1,234.56)"
    else:
        result["amount_format"] = "Unknown"

    # Get all lines with annotations
    all_lines = []
    for pg_idx, page_text in enumerate(pages[:3]):
        for line in page_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line_type = "normal"
            for _, pattern, _ in DATE_PATTERNS:
                if re.match(pattern, line) or re.search(r"^\d{2}-[A-Za-z]{3}-\d{4}", line):
                    line_type = "date"
                    break
            if re.search(r"[\d,]+\.\d{2}", line) and len(re.findall(r"[\d,]+\.\d{2}", line)) >= 2:
                line_type = "transaction"
            all_lines.append({
                "page": pg_idx + 1,
                "text": line,
                "type": line_type,
            })

    result["all_lines"] = all_lines[:100]  # First 100 lines

    # Extract keywords
    keywords = []
    if result["swift"]:
        keywords.append(result["swift"])
    if result["bank_name"] != "Unknown Bank":
        keywords.append(result["bank_name"])
    result["keywords"] = keywords

    return result


# ── HOME ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    # List existing profiles
    profiles = []
    for f in sorted(PROFILE_DIR.glob("*.json")):
        try:
            p = json.loads(f.read_text())
            profiles.append(p)
        except: pass

    profile_html = ""
    if profiles:
        profile_html = '<div class="card"><div class="card-title">Trained Bank Profiles (' + str(len(profiles)) + ')</div>'
        for p in profiles:
            profile_html += f"""
            <div class="profile-card" style="margin-bottom:8px">
              <div>
                <strong>{p.get("bank_name","?")}</strong>
                <span class="badge badge-blue" style="margin-left:8px">{p.get("swift","?")}</span>
                <span class="badge badge-ok" style="margin-left:4px">{p.get("currency","?")}</span>
                <span style="color:#64748b;font-size:12px;margin-left:8px">
                  Date: {p.get("date_pattern","?")} · 
                  Cols: {p.get("col_balance","?")} balance
                </span>
              </div>
              <div style="display:flex;gap:8px">
                <a href="/trainer/test/{p.get("swift","")}" class="btn btn-ghost" 
                   style="padding:6px 12px;font-size:12px">Test</a>
                <button onclick="deleteProfile('{p.get("swift","")}')" 
                        class="btn btn-danger" style="padding:6px 12px;font-size:12px">Delete</button>
              </div>
            </div>"""
        profile_html += '</div>'

    body = """
    <h1 style="font-size:20px;font-weight:700;margin-bottom:6px">Bank Format Trainer</h1>
    <p style="font-size:13px;color:#64748b;margin-bottom:20px">
      Upload any bank statement PDF to automatically detect its format and add parser support.
    </p>

    """ + profile_html + """

    <div class="card">
      <div class="card-title">Upload New Bank Statement PDF</div>
      <div class="upload-zone" id="zone" onclick="document.getElementById('inp').click()">
        <div style="font-size:32px;margin-bottom:10px">&#128196;</div>
        <div style="font-size:16px;font-weight:600;margin-bottom:6px">
          Drop any bank statement PDF here
        </div>
        <div style="font-size:13px;color:#64748b">
          ABA, Wing, ACLEDA, Prince, Vattanac, Canadia, or any other bank
        </div>
        <input type="file" id="inp" accept=".pdf" style="display:none" onchange="upload(this)">
      </div>
      <div id="status" style="margin-top:12px;font-size:13px;color:#64748b"></div>
    </div>

    <script>
    function upload(inp) {
      var f = inp.files[0]; if (!f) return;
      document.getElementById('status').textContent = 'Analysing ' + f.name + '...';
      var fd = new FormData();
      fd.append('pdf_file', inp.files[0]);
      fetch('/trainer/analyse', {method:'POST', body:fd})
        .then(function(r) { return r.json(); })
        .then(function(d) {
          if (d.uid) window.location.href = '/trainer/review/' + d.uid;
          else document.getElementById('status').textContent = 'Error: ' + (d.error||'unknown');
        })
        .catch(function(e) { document.getElementById('status').textContent = 'Failed: ' + e; });
    }
    function deleteProfile(swift) {
      if (!confirm('Delete profile for ' + swift + '?')) return;
      fetch('/trainer/profile/' + swift, {method:'DELETE'})
        .then(function() { location.reload(); });
    }
    var zone = document.getElementById('zone');
    zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.style.borderColor='#2563eb'; });
    zone.addEventListener('dragleave', function() { zone.style.borderColor=''; });
    zone.addEventListener('drop', function(e) {
      e.preventDefault(); zone.style.borderColor='';
      document.getElementById('inp').files = e.dataTransfer.files;
      upload(document.getElementById('inp'));
    });
    </script>"""
    return HTMLResponse(page("Upload", body))


# ── ANALYSE ───────────────────────────────────────────────────────────────────

@app.post("/analyse")
async def analyse(pdf_file: UploadFile = File(...)):
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    uid      = str(uuid.uuid4())[:8]
    pdf_path = TRAIN_DIR / (uid + "_" + pdf_file.filename)
    raw      = await pdf_file.read()
    pdf_path.write_bytes(raw)

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages[:5]:
            pages.append(p.extract_text(x_tolerance=2, y_tolerance=3) or "")

    full_text = "\n".join(pages)
    detected  = auto_detect(full_text, pages)
    detected["uid"]      = uid
    detected["filename"] = pdf_file.filename
    detected["pdf_path"] = str(pdf_path)

    (TRAIN_DIR / (uid + "_analysis.json")).write_text(json.dumps(detected))
    return JSONResponse({"uid": uid})


# ── REVIEW ────────────────────────────────────────────────────────────────────

@app.get("/review/{uid}", response_class=HTMLResponse)
def review(uid: str):
    af = TRAIN_DIR / (uid + "_analysis.json")
    if not af.exists():
        return HTMLResponse(page("Error", '<div class="alert alert-err">Analysis not found</div>'))

    det = json.loads(af.read_text())

    # Build annotated raw text
    lines_html = ""
    for i, line in enumerate(det["all_lines"]):
        cls   = ""
        color = "#e2e8f0"
        if line["type"] == "date":
            color = "#4ade80"
            cls   = "line-date"
        elif line["type"] == "transaction":
            color = "#fbbf24"
        text_escaped = line["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        lines_html += f'<div><span class="line-num">{i+1:3d}</span><span style="color:{color}">{text_escaped}</span></div>'

    # Check if profile already exists
    existing = ""
    prof_file = PROFILE_DIR / (det["swift"] + ".json")
    if det["swift"] and prof_file.exists():
        existing = '<div class="alert alert-warn">⚠ A profile for ' + det["swift"] + ' already exists. Saving will overwrite it.</div>'

    body = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <div>
        <h1 style="font-size:20px;font-weight:700">Review Detected Format</h1>
        <p style="font-size:13px;color:#64748b;margin-top:3px">{det["filename"]}</p>
      </div>
      <a href="/trainer/" class="btn btn-ghost">Upload Another</a>
    </div>

    {existing}

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px">
      <div>
        <div class="card">
          <div class="card-title">Auto-Detected Information</div>
          <table>
            <tr><td style="font-weight:600;width:140px">Bank Name</td>
                <td><strong>{det["bank_name"]}</strong></td></tr>
            <tr><td style="font-weight:600">SWIFT Code</td>
                <td><span class="badge badge-blue">{det["swift"] or "Not found"}</span></td></tr>
            <tr><td style="font-weight:600">Currency</td>
                <td><span class="badge badge-ok">{det["currency"]}</span>
                    {"<span class='badge badge-warn' style='margin-left:4px'>KHR</span>" if det["has_khr"] else ""}
                    {"<span class='badge badge-ok' style='margin-left:4px'>USD</span>" if det["has_usd"] else ""}
                </td></tr>
            <tr><td style="font-weight:600">Date Format</td>
                <td>{det["date_pattern"] or "Unknown"} 
                    <span style="color:#64748b">({det["date_example"]})</span></td></tr>
            <tr><td style="font-weight:600">Amount Format</td>
                <td>{det["amount_format"]}</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="card-title">Configure Bank Profile</div>
          <form id="profile-form">
            <div class="form-group">
              <label>Bank Name *</label>
              <input type="text" id="bank_name" value="{det["bank_name"]}" required>
            </div>
            <div class="grid-2">
              <div class="form-group">
                <label>SWIFT Code *</label>
                <input type="text" id="swift" value="{det["swift"]}" required>
              </div>
              <div class="form-group">
                <label>Currency</label>
                <select id="currency">
                  <option value="USD" {"selected" if det["currency"]=="USD" else ""}>USD</option>
                  <option value="KHR" {"selected" if det["currency"]=="KHR" else ""}>KHR</option>
                  <option value="BOTH">Both (USD + KHR)</option>
                </select>
              </div>
            </div>
            <div class="form-group">
              <label>Date Pattern</label>
              <select id="date_pattern">
                <option value="DD-Mon-YYYY" {"selected" if det["date_pattern"]=="DD-Mon-YYYY" else ""}>DD-Mon-YYYY (22-Apr-2026)</option>
                <option value="Mon DD, YYYY" {"selected" if det["date_pattern"]=="Mon DD, YYYY" else ""}>Mon DD, YYYY (Aug 01, 2025)</option>
                <option value="DD MON YY" {"selected" if det["date_pattern"]=="DD MON YY" else ""}>DD MON YY (24 MAR 23)</option>
                <option value="DD/MM/YYYY" {"selected" if det["date_pattern"]=="DD/MM/YYYY" else ""}>DD/MM/YYYY (01/02/2025)</option>
                <option value="DD/MM/YY" {"selected" if det["date_pattern"]=="DD/MM/YY" else ""}>DD/MM/YY (01/02/23)</option>
                <option value="YYYY-MM-DD" {"selected" if det["date_pattern"]=="YYYY-MM-DD" else ""}>YYYY-MM-DD (2025-08-01)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Column Order (left to right)</label>
              <select id="col_order">
                <option value="date_desc_debit_credit_balance">Date | Desc | Debit | Credit | Balance</option>
                <option value="date_ref_desc_debit_credit_balance">Date | Ref | Desc | Debit | Credit | Balance</option>
                <option value="date_branch_ref_desc_debit_credit_balance">Date | Branch | Ref | Desc | Debit | Credit | Balance</option>
                <option value="date_desc_ref_date_debit_credit_balance">Date | Desc | Ref | Date | Debit | Credit | Balance (ACLEDA)</option>
                <option value="date_desc_credit_debit_balance">Date | Desc | Credit | Debit | Balance</option>
              </select>
            </div>
            <div class="grid-2">
              <div class="form-group">
                <label>Empty Column Marker</label>
                <select id="empty_marker">
                  <option value="-">Dash (-)</option>
                  <option value="blank">Blank/Empty</option>
                  <option value="0">Zero (0)</option>
                </select>
              </div>
              <div class="form-group">
                <label>Date Location</label>
                <select id="date_location">
                  <option value="start_of_line">Start of transaction line</option>
                  <option value="own_line">Own line (before amounts)</option>
                  <option value="own_line_with_desc">Own line + desc on same line</option>
                </select>
              </div>
            </div>
            <div class="form-group">
              <label>Direction Detection Keywords (comma separated)</label>
              <input type="text" id="debit_keywords" 
                     placeholder="e.g. transfer to,payment to,fee charge,withdrawal,debit"
                     value="transfer to,payment to,fee,charge,withdrawal,debit,qr payment debit">
            </div>
            <div class="form-group">
              <label>Skip Lines Containing (comma separated)</label>
              <input type="text" id="skip_keywords"
                     value="page ,balance at period,this is a computer,swift code,statement period">
            </div>
            <div class="form-group">
              <label>Notes / Remarks</label>
              <textarea id="notes" rows="2" placeholder="Any special handling notes..."></textarea>
            </div>
            <div style="display:flex;gap:10px;margin-top:16px">
              <button type="button" class="btn btn-primary btn-lg" onclick="saveProfile('{uid}')">
                Save Bank Profile
              </button>
              <button type="button" class="btn btn-ghost" onclick="testParse('{uid}')">
                Test Parse
              </button>
            </div>
          </form>
        </div>
      </div>

      <div>
        <div class="card">
          <div class="card-title">
            Raw PDF Text 
            <span style="font-weight:400;color:#64748b;font-size:10px;margin-left:8px">
              &#x1F7E2; = date line &nbsp; &#x1F7E1; = transaction line
            </span>
          </div>
          <div class="raw-text" id="raw-text">{lines_html}</div>
        </div>

        <div id="parse-result" style="display:none">
          <div class="card">
            <div class="card-title">Test Parse Results</div>
            <div id="parse-content"></div>
          </div>
        </div>
      </div>
    </div>

    <script>
    function getProfile() {{
      return {{
        uid:           '{uid}',
        bank_name:     document.getElementById('bank_name').value,
        swift:         document.getElementById('swift').value.toUpperCase(),
        currency:      document.getElementById('currency').value,
        date_pattern:  document.getElementById('date_pattern').value,
        date_location: document.getElementById('date_location').value,
        col_order:     document.getElementById('col_order').value,
        empty_marker:  document.getElementById('empty_marker').value,
        debit_keywords:document.getElementById('debit_keywords').value,
        skip_keywords: document.getElementById('skip_keywords').value,
        notes:         document.getElementById('notes').value,
        filename:      '{det["filename"]}',
        created_at:    new Date().toISOString(),
      }};
    }}

    function saveProfile(uid) {{
      var p = getProfile();
      if (!p.bank_name || !p.swift) {{ alert('Bank name and SWIFT code required'); return; }}
      fetch('/trainer/save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(p)
      }})
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (d.ok) {{
          alert('Profile saved for ' + p.bank_name + '!');
          window.location.href = '/trainer/';
        }} else {{
          alert('Error: ' + d.error);
        }}
      }});
    }}

    function testParse(uid) {{
      var p = getProfile();
      document.getElementById('parse-result').style.display = 'block';
      document.getElementById('parse-content').innerHTML = 'Parsing...';
      fetch('/trainer/test-parse', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(p)
      }})
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (d.error) {{
          document.getElementById('parse-content').innerHTML =
            '<div class="alert alert-err">' + d.error + '</div>';
          return;
        }}
        var html = '<div class="alert alert-ok">Found ' + d.count + ' transactions</div>';
        html += '<table><thead><tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr></thead><tbody>';
        (d.transactions||[]).slice(0,20).forEach(function(t) {{
          html += '<tr><td>' + t.date + '</td><td>' + t.desc + '</td>' +
                  '<td style="color:#dc2626">' + (t.money_out > 0 ? t.money_out.toFixed(2) : '') + '</td>' +
                  '<td style="color:#059669">' + (t.money_in  > 0 ? t.money_in.toFixed(2)  : '') + '</td>' +
                  '<td>' + t.balance.toFixed(2) + '</td></tr>';
        }});
        html += '</tbody></table>';
        if (d.count > 20) html += '<p style="color:#64748b;padding:8px;font-size:12px">Showing 20 of ' + d.count + ' transactions</p>';
        document.getElementById('parse-content').innerHTML = html;
      }});
    }}
    </script>"""
    return HTMLResponse(page("Review", body))


# ── SAVE PROFILE ──────────────────────────────────────────────────────────────

@app.post("/save")
async def save_profile(request_body: dict):
    try:
        profile  = request_body
        swift    = profile.get("swift","").upper()
        if not swift:
            return JSONResponse({"error": "SWIFT code required"})

        prof_path = PROFILE_DIR / (swift + ".json")
        prof_path.write_text(json.dumps(profile, indent=2))
        log.info(f"Saved bank profile: {swift} - {profile.get('bank_name')}")
        return JSONResponse({"ok": True, "swift": swift})
    except Exception as e:
        return JSONResponse({"error": str(e)})

from fastapi import Request

@app.post("/save")
async def save_profile(request: Request):
    try:
        profile  = await request.json()
        swift    = profile.get("swift","").upper()
        if not swift:
            return JSONResponse({"error": "SWIFT code required"})
        prof_path = PROFILE_DIR / (swift + ".json")
        prof_path.write_text(json.dumps(profile, indent=2))
        return JSONResponse({"ok": True, "swift": swift})
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ── TEST PARSE ────────────────────────────────────────────────────────────────

@app.post("/test-parse")
async def test_parse(request: Request):
    """Try to parse the PDF using the configured profile."""
    try:
        profile  = await request.json()
        uid      = profile.get("uid","")
        af       = TRAIN_DIR / (uid + "_analysis.json")
        if not af.exists():
            return JSONResponse({"error": "Analysis not found"})

        det      = json.loads(af.read_text())
        pdf_path = det["pdf_path"]

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text(x_tolerance=2, y_tolerance=3) or "")

        txns = _parse_with_profile(pages, profile)
        return JSONResponse({
            "count":        len(txns),
            "transactions": txns[:50],
        })
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e) + "\n" + traceback.format_exc()})


def _parse_with_profile(pages: list, profile: dict) -> list:
    """Generic parser using a bank profile configuration."""
    transactions = []
    date_pattern = profile.get("date_pattern","")
    col_order    = profile.get("col_order","")
    debit_kws    = [x.strip().lower() for x in profile.get("debit_keywords","").split(",") if x.strip()]
    skip_kws     = [x.strip().lower() for x in profile.get("skip_keywords","").split(",") if x.strip()]
    date_loc     = profile.get("date_location","start_of_line")
    currency     = profile.get("currency","USD")

    # Build date regex
    date_regexes = {
        "DD-Mon-YYYY":  r"(\d{2}-[A-Za-z]{3}-\d{4})",
        "Mon DD, YYYY": r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})",
        "DD MON YY":    r"(\d{1,2}\s+[A-Z]{3}\s+\d{2})",
        "DD/MM/YYYY":   r"(\d{2}/\d{2}/\d{4})",
        "DD/MM/YY":     r"(\d{2}/\d{2}/\d{2})",
        "YYYY-MM-DD":   r"(\d{4}-\d{2}-\d{2})",
    }
    date_re = re.compile(date_regexes.get(date_pattern, r"(\d{2}-\d{2}-\d{4})"))
    amt_re  = re.compile(r"([\d,]+\.?\d*)")

    MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                 "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

    def parse_date(s):
        s = s.strip()
        if date_pattern == "DD-Mon-YYYY":
            m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{4})", s)
            if m:
                mon = MONTH_MAP.get(m.group(2).lower())
                if mon: return date(int(m.group(3)), mon, int(m.group(1))).isoformat()
        elif date_pattern == "Mon DD, YYYY":
            m = re.match(r"([A-Za-z]{3})\s+(\d{1,2}),?\s+(\d{4})", s)
            if m:
                mon = MONTH_MAP.get(m.group(1).lower())
                if mon: return date(int(m.group(3)), mon, int(m.group(2))).isoformat()
        elif date_pattern == "DD MON YY":
            m = re.match(r"(\d{1,2})\s+([A-Z]{3})\s+(\d{2})", s)
            if m:
                mon = MONTH_MAP.get(m.group(2).lower())
                yr  = int(m.group(3)) + 2000
                if mon: return date(yr, mon, int(m.group(1))).isoformat()
        elif date_pattern in ("DD/MM/YYYY","DD/MM/YY"):
            m = re.match(r"(\d{2})/(\d{2})/(\d{2,4})", s)
            if m:
                yr = int(m.group(3))
                if yr < 100: yr += 2000
                return date(yr, int(m.group(2)), int(m.group(1))).isoformat()
        elif date_pattern == "YYYY-MM-DD":
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
            if m: return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        return None

    if date_loc == "start_of_line":
        # Date is at start of each transaction line (ABA, ACLEDA style)
        for page_text in pages:
            lines = [l.strip() for l in page_text.split("\n")]
            i = 0
            while i < len(lines):
                line = lines[i]
                if any(sk in line.lower() for sk in skip_kws):
                    i += 1; continue

                dm = date_re.match(line)
                if dm:
                    trx_date = parse_date(dm.group(1))
                    rest     = line[dm.end():].strip()
                    # Collect continuation
                    j = i + 1
                    while j < len(lines):
                        nl = lines[j]
                        if date_re.match(nl): break
                        if any(sk in nl.lower() for sk in skip_kws): break
                        rest += " " + nl
                        j += 1
                    i = j

                    # Parse amounts
                    amounts = [float(m.group(1).replace(",",""))
                               for m in amt_re.finditer(rest)
                               if float(m.group(1).replace(",","")) > 0
                               and "." in m.group(1)]
                    if len(amounts) < 2: continue

                    balance = amounts[-1]
                    amount  = amounts[-2]
                    rl      = rest.lower()
                    is_debit = any(k in rl for k in debit_kws)

                    desc = re.sub(r"[\d,]+\.?\d*", "", rest)
                    desc = re.sub(r"\s+", " ", desc).strip()[:100]

                    if trx_date and amount > 0:
                        transactions.append({
                            "date":      trx_date,
                            "desc":      desc,
                            "money_in":  round(amount, 2) if not is_debit else 0.0,
                            "money_out": round(amount, 2) if is_debit else 0.0,
                            "balance":   round(balance, 2),
                        })
                else:
                    i += 1

    else:
        # Date on own line (Wing Bank style)
        for page_text in pages:
            lines = [l.strip() for l in page_text.split("\n")]
            i = 0
            prev_desc = ""
            while i < len(lines):
                line = lines[i]
                if any(sk in line.lower() for sk in skip_kws):
                    prev_desc = ""; i += 1; continue

                dm = date_re.search(line)
                if dm:
                    trx_date    = parse_date(dm.group(1))
                    inline_desc = line[dm.end():].strip()
                    i += 1
                    body_parts  = []
                    while i < len(lines):
                        nl = lines[i]
                        if date_re.search(nl): break
                        if any(sk in nl.lower() for sk in skip_kws): break
                        body_parts.append(nl)
                        i += 1
                    body = " ".join(body_parts)
                    amounts = [float(m.group(1).replace(",",""))
                               for m in amt_re.finditer(body)
                               if "." in m.group(1) and float(m.group(1).replace(",","")) > 0]
                    if len(amounts) < 2:
                        prev_desc = inline_desc or prev_desc; continue
                    balance = amounts[-1]; amount = amounts[-2]
                    rl = (inline_desc + " " + body).lower()
                    is_debit = any(k in rl for k in debit_kws)
                    desc = (prev_desc + " " + inline_desc + " " + body)
                    desc = re.sub(r"[\d,]+\.?\d*", "", desc)
                    desc = re.sub(r"\s+", " ", desc).strip()[:100]
                    if trx_date and amount > 0:
                        transactions.append({
                            "date":      trx_date,
                            "desc":      desc,
                            "money_in":  round(amount, 2) if not is_debit else 0.0,
                            "money_out": round(amount, 2) if is_debit else 0.0,
                            "balance":   round(balance, 2),
                        })
                    prev_desc = ""
                elif len(line) > 3:
                    prev_desc = line; i += 1
                else:
                    i += 1

    return transactions


# ── PROFILES LIST ─────────────────────────────────────────────────────────────

@app.get("/profiles", response_class=HTMLResponse)
def profiles():
    profs = []
    for f in sorted(PROFILE_DIR.glob("*.json")):
        try:
            profs.append(json.loads(f.read_text()))
        except: pass

    rows = ""
    for p in profs:
        rows += f"""<tr>
          <td><strong>{p.get("bank_name","?")}</strong></td>
          <td><span class="badge badge-blue">{p.get("swift","?")}</span></td>
          <td><span class="badge badge-ok">{p.get("currency","?")}</span></td>
          <td>{p.get("date_pattern","?")}</td>
          <td>{p.get("col_order","?")}</td>
          <td>{p.get("created_at","")[:10]}</td>
          <td>
            <button onclick="del('{p.get("swift","")}')" 
                    class="btn btn-danger" style="padding:4px 10px;font-size:11px">Delete</button>
          </td>
        </tr>"""

    body = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <h1 style="font-size:20px;font-weight:700">Bank Profiles ({len(profs)})</h1>
      <a href="/trainer/" class="btn btn-primary">+ Add New</a>
    </div>
    <div class="card">
      <table>
        <thead><tr>
          <th>Bank Name</th><th>SWIFT</th><th>Currency</th>
          <th>Date Pattern</th><th>Column Order</th><th>Created</th><th></th>
        </tr></thead>
        <tbody>{rows if rows else "<tr><td colspan='7' style='text-align:center;color:#64748b;padding:20px'>No profiles yet — upload a PDF to train</td></tr>"}</tbody>
      </table>
    </div>
    <script>
    function del(swift) {{
      if (!confirm('Delete ' + swift + '?')) return;
      fetch('/trainer/profile/' + swift, {{method:'DELETE'}})
        .then(function() {{ location.reload(); }});
    }}
    </script>"""
    return HTMLResponse(page("Profiles", body))


@app.delete("/profile/{swift}")
def delete_profile(swift: str):
    f = PROFILE_DIR / (swift.upper() + ".json")
    if f.exists():
        f.unlink()
    return JSONResponse({"ok": True})


# ── API: GET ALL PROFILES ─────────────────────────────────────────────────────

@app.get("/api/profiles")
def get_profiles():
    """Return all saved profiles as JSON — used by coho_extractor."""
    profs = []
    for f in PROFILE_DIR.glob("*.json"):
        try:
            profs.append(json.loads(f.read_text()))
        except: pass
    return JSONResponse(profs)


@app.get("/health")
def health():
    return {"status": "ok", "profiles": len(list(PROFILE_DIR.glob("*.json")))}
