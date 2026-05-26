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
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    body = """
    <h1 style="font-size:20px;font-weight:700;margin-bottom:20px">Bank Statement Analyser</h1>
    <div class="card">
      <div class="card-title">Upload COHO — Bank Statement Analyser</div>
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
    banks_html = '<div class="card" style="margin-top:18px"><div class="card-title">Supported Financial Institutions</div><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-top:4px"><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏦</span><div><div style="font-weight:600;color:#1F4E79">ABA Bank</div><div style="color:#94a3b8;font-size:10px">USD / KHR</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🦅</span><div><div style="font-weight:600;color:#1F4E79">Wing Bank</div><div style="color:#94a3b8;font-size:10px">USD / KHR</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏛</span><div><div style="font-weight:600;color:#1F4E79">ACLEDA Bank</div><div style="color:#94a3b8;font-size:10px">KHR</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">💳</span><div><div style="font-weight:600;color:#1F4E79">KB Prasac</div><div style="color:#94a3b8;font-size:10px">USD / KHR</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏦</span><div><div style="font-weight:600;color:#1F4E79">Philip Bank</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🌾</span><div><div style="font-weight:600;color:#1F4E79">AMRET MFI</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏢</span><div><div style="font-weight:600;color:#1F4E79">Woori Bank</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">💰</span><div><div style="font-weight:600;color:#1F4E79">Hattha Bank</div><div style="color:#94a3b8;font-size:10px">USD / KHR</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏦</span><div><div style="font-weight:600;color:#1F4E79">Canadia Bank</div><div style="color:#94a3b8;font-size:10px">KHR</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏛</span><div><div style="font-weight:600;color:#1F4E79">Sathapana Bank</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">📮</span><div><div style="font-weight:600;color:#1F4E79">Post Bank</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏦</span><div><div style="font-weight:600;color:#1F4E79">Maybank</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🌱</span><div><div style="font-weight:600;color:#1F4E79">AMK Microfinance</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏦</span><div><div style="font-weight:600;color:#1F4E79">Taiwan Coop Bank</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div><div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px"><span style="font-size:16px">🏦</span><div><div style="font-weight:600;color:#1F4E79">LOLC Cambodia</div><div style="color:#94a3b8;font-size:10px">USD</div></div></div></div><div style="margin-top:10px;font-size:11px;color:#94a3b8">Auto-detected - no configuration needed | New banks via Bank Trainer</div></div>'
    body += banks_html
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
    return JSONResponse(
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
