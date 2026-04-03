"""
dashboard.py — Document Fraud Detection Dashboard
Serves a full analytics dashboard from PostgreSQL data.
Mounted at /dashboard/ via Nginx.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("fraud_detect.dashboard")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI(title="Fraud Detection Dashboard", version="1.0.0")


# ── Data helpers ───────────────────────────────────────────────────────────────

def _query(sql: str, params: dict = {}):
    db = SessionLocal()
    try:
        result = db.execute(text(sql), params)
        return [dict(row._mapping) for row in result]
    except Exception as e:
        log.error(f"Query failed: {e}")
        return []
    finally:
        db.close()


def _get_stats(days: int = 30):
    since = datetime.utcnow() - timedelta(days=days)
    rows  = _query("""
        SELECT
            COUNT(*)                                          AS total,
            COUNT(*) FILTER (WHERE risk_level = 'HIGH')      AS high_risk,
            COUNT(*) FILTER (WHERE risk_level = 'MEDIUM')    AS medium_risk,
            COUNT(*) FILTER (WHERE risk_level = 'LOW')       AS low_risk,
            COUNT(*) FILTER (WHERE doc_type = 'pdf')         AS pdfs,
            COUNT(*) FILTER (WHERE doc_type = 'image')       AS images,
            COUNT(*) FILTER (WHERE doc_type = 'id_card')     AS id_cards,
            ROUND(AVG(risk_score), 1)                        AS avg_score,
            COUNT(DISTINCT applicant_id)
              FILTER (WHERE applicant_id IS NOT NULL)         AS unique_applicants
        FROM screening_logs
        WHERE screened_at >= :since
    """, {"since": since})
    return rows[0] if rows else {}


def _get_recent(limit: int = 20):
    return _query("""
        SELECT id, file_name, doc_type, risk_level, risk_score,
               flags, screened_at, applicant_id, id_number
        FROM screening_logs
        ORDER BY screened_at DESC
        LIMIT :limit
    """, {"limit": limit})


def _get_daily_trend(days: int = 14):
    since = datetime.utcnow() - timedelta(days=days)
    return _query("""
        SELECT
            DATE(screened_at)                                    AS day,
            COUNT(*)                                             AS total,
            COUNT(*) FILTER (WHERE risk_level = 'HIGH')         AS high,
            COUNT(*) FILTER (WHERE risk_level = 'MEDIUM')       AS medium,
            COUNT(*) FILTER (WHERE risk_level = 'LOW')          AS low
        FROM screening_logs
        WHERE screened_at >= :since
        GROUP BY DATE(screened_at)
        ORDER BY day ASC
    """, {"since": since})


def _get_top_flags(days: int = 30, limit: int = 10):
    since = datetime.utcnow() - timedelta(days=days)
    rows  = _query("""
        SELECT flags, screened_at FROM screening_logs
        WHERE screened_at >= :since
    """, {"since": since})

    counts = {}
    for row in rows:
        flags = row.get("flags") or []
        if isinstance(flags, str):
            try: flags = json.loads(flags)
            except: flags = []
        for f in flags:
            f = f.split(":")[0]   # strip dynamic parts e.g. suspicious_tool:photoshop
            counts[f] = counts.get(f, 0) + 1

    sorted_flags = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [{"flag": f, "count": c} for f, c in sorted_flags]


def _get_applicant_history(applicant_id: str):
    return _query("""
        SELECT id, file_name, doc_type, risk_level, risk_score,
               flags, screened_at, id_number
        FROM screening_logs
        WHERE applicant_id = :aid
        ORDER BY screened_at DESC
    """, {"aid": applicant_id})


def _risk_color(level):
    return {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(level, "#94a3b8")

def _action_badge(level):
    action = {"HIGH": "REJECT", "MEDIUM": "REVIEW", "LOW": "PASS"}.get(level, "—")
    color  = {"REJECT": "#ef4444", "REVIEW": "#f59e0b", "PASS": "#22c55e"}.get(action, "#94a3b8")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700">{action}</span>'


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats(days: int = 30):
    return JSONResponse(_get_stats(days))

@app.get("/api/recent")
def api_recent(limit: int = 20):
    rows = _get_recent(limit)
    for r in rows:
        if r.get("screened_at"):
            r["screened_at"] = r["screened_at"].isoformat()
    return JSONResponse(rows)

@app.get("/api/trend")
def api_trend(days: int = 14):
    rows = _get_daily_trend(days)
    for r in rows:
        if r.get("day"):
            r["day"] = str(r["day"])
    return JSONResponse(rows)

@app.get("/api/flags")
def api_flags(days: int = 30):
    return JSONResponse(_get_top_flags(days))

@app.get("/api/applicant/{applicant_id}")
def api_applicant(applicant_id: str):
    rows = _get_applicant_history(applicant_id)
    for r in rows:
        if r.get("screened_at"):
            r["screened_at"] = r["screened_at"].isoformat()
    return JSONResponse(rows)


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    stats  = _get_stats(30)
    recent = _get_recent(15)
    flags  = _get_top_flags(30, 8)
    trend  = _get_daily_trend(14)

    # ── Recent submissions table rows ─────────────────────────────────────────
    rows_html = ""
    for r in recent:
        level     = r.get("risk_level", "—")
        score     = r.get("risk_score", "—")
        fname     = r.get("file_name", "—")
        dtype     = r.get("doc_type", "—").upper()
        applicant = r.get("applicant_id") or "—"
        id_num    = r.get("id_number") or "—"
        screened  = str(r.get("screened_at", ""))[:19].replace("T", " ")
        flag_list = r.get("flags") or []
        if isinstance(flag_list, str):
            try: flag_list = json.loads(flag_list)
            except: flag_list = []
        flag_count = len(flag_list)
        color      = _risk_color(level)

        rows_html += f"""
        <tr>
          <td>{screened}</td>
          <td title="{fname}">{fname[:25]}{"…" if len(fname)>25 else ""}</td>
          <td><span style="background:#f1f5f9;padding:2px 8px;border-radius:3px;font-size:11px">{dtype}</span></td>
          <td>{applicant}</td>
          <td style="font-family:monospace;font-size:11px">{id_num[:12] if id_num != "—" else "—"}</td>
          <td style="font-weight:700;color:{color}">{level}</td>
          <td style="text-align:center">{score}</td>
          <td style="text-align:center">{_action_badge(level)}</td>
          <td style="text-align:center;color:#64748b">{flag_count}</td>
        </tr>"""

    # ── Top flags bars ────────────────────────────────────────────────────────
    max_count  = max((f["count"] for f in flags), default=1)
    flags_html = ""
    for f in flags:
        pct = int(f["count"] / max_count * 100)
        flags_html += f"""
        <div style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
            <span style="color:#334155;font-family:monospace">{f["flag"]}</span>
            <span style="color:#64748b;font-weight:600">{f["count"]}</span>
          </div>
          <div style="background:#f1f5f9;border-radius:4px;height:8px">
            <div style="background:#3b82f6;width:{pct}%;height:8px;border-radius:4px"></div>
          </div>
        </div>"""

    # ── Trend sparkline data ──────────────────────────────────────────────────
    trend_labels = json.dumps([str(r.get("day",""))[-5:] for r in trend])
    trend_high   = json.dumps([r.get("high", 0) for r in trend])
    trend_med    = json.dumps([r.get("medium", 0) for r in trend])
    trend_low    = json.dumps([r.get("low", 0) for r in trend])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fraud Detection Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#0f172a}}
    .topbar{{background:#0f172a;color:#fff;padding:16px 32px;display:flex;align-items:center;justify-content:space-between}}
    .topbar h1{{font-size:16px;font-weight:600;letter-spacing:.5px}}
    .topbar p{{font-size:12px;color:#94a3b8;margin-top:2px}}
    .badge{{background:#1e293b;border-radius:4px;padding:4px 12px;font-size:12px;color:#94a3b8}}
    .main{{padding:24px 32px}}
    .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:24px}}
    .stat{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px 20px}}
    .stat .num{{font-size:26px;font-weight:700;margin-bottom:2px}}
    .stat .lbl{{font-size:12px;color:#64748b}}
    .row{{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-bottom:24px}}
    .card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:20px}}
    .card h3{{font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
              letter-spacing:.5px;margin-bottom:16px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th{{padding:10px 8px;text-align:left;font-size:11px;font-weight:600;
        color:#64748b;border-bottom:2px solid #f1f5f9;text-transform:uppercase}}
    td{{padding:10px 8px;border-bottom:1px solid #f8fafc;vertical-align:middle}}
    tr:hover td{{background:#f8fafc}}
    .search-bar{{display:flex;gap:10px;margin-bottom:20px}}
    .search-bar input{{flex:1;padding:9px 14px;border:1px solid #e2e8f0;border-radius:6px;
                       font-size:13px;outline:none}}
    .search-bar button{{padding:9px 18px;background:#3b82f6;color:#fff;border:none;
                        border-radius:6px;cursor:pointer;font-size:13px;font-weight:500}}
    .search-bar button:hover{{background:#2563eb}}
    #applicant-result{{margin-top:16px}}
  </style>
</head>
<body>

<div class="topbar">
  <div>
    <h1>🔍 Document Fraud Detection Dashboard</h1>
    <p>Cambodia KYC / Onboarding — Last 30 days</p>
  </div>
  <span class="badge">Auto-refresh off · <a href="javascript:location.reload()" style="color:#60a5fa;text-decoration:none">Refresh</a></span>
</div>

<div class="main">

  <!-- Stats -->
  <div class="stats">
    <div class="stat">
      <div class="num">{stats.get("total", 0)}</div>
      <div class="lbl">Total Screened</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#ef4444">{stats.get("high_risk", 0)}</div>
      <div class="lbl">HIGH Risk</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#f59e0b">{stats.get("medium_risk", 0)}</div>
      <div class="lbl">MEDIUM Risk</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#22c55e">{stats.get("low_risk", 0)}</div>
      <div class="lbl">LOW Risk</div>
    </div>
    <div class="stat">
      <div class="num">{stats.get("id_cards", 0)}</div>
      <div class="lbl">ID Cards</div>
    </div>
    <div class="stat">
      <div class="num">{stats.get("pdfs", 0)}</div>
      <div class="lbl">PDFs</div>
    </div>
    <div class="stat">
      <div class="num">{stats.get("avg_score", 0)}</div>
      <div class="lbl">Avg Risk Score</div>
    </div>
    <div class="stat">
      <div class="num">{stats.get("unique_applicants", 0)}</div>
      <div class="lbl">Unique Applicants</div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="row">
    <div class="card">
      <h3>Risk Trend — Last 14 Days</h3>
      <canvas id="trendChart" height="120"></canvas>
    </div>
    <div class="card">
      <h3>Top Fraud Flags</h3>
      {flags_html if flags_html else '<p style="color:#94a3b8;font-size:13px">No flags yet</p>'}
    </div>
  </div>

  <!-- Applicant lookup -->
  <div class="card" style="margin-bottom:20px">
    <h3>Applicant Case Lookup</h3>
    <div class="search-bar">
      <input type="text" id="applicant-input" placeholder="Enter Applicant ID or ID Number…">
      <button onclick="lookupApplicant()">Search</button>
    </div>
    <div id="applicant-result"></div>
  </div>

  <!-- Recent submissions -->
  <div class="card">
    <h3>Recent Submissions</h3>
    <table>
      <thead>
        <tr>
          <th>Screened At</th>
          <th>File</th>
          <th>Type</th>
          <th>Applicant</th>
          <th>ID Number</th>
          <th>Risk</th>
          <th>Score</th>
          <th>Action</th>
          <th>Flags</th>
        </tr>
      </thead>
      <tbody>
        {rows_html if rows_html else
         '<tr><td colspan="9" style="text-align:center;padding:30px;color:#94a3b8">No submissions yet</td></tr>'}
      </tbody>
    </table>
  </div>

</div>

<script>
// ── Trend chart ───────────────────────────────────────────────────────────────
const ctx = document.getElementById('trendChart').getContext('2d');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {trend_labels},
    datasets: [
      {{ label: 'HIGH',   data: {trend_high}, backgroundColor: '#ef4444' }},
      {{ label: 'MEDIUM', data: {trend_med},  backgroundColor: '#f59e0b' }},
      {{ label: 'LOW',    data: {trend_low},  backgroundColor: '#22c55e' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ stacked: true, beginAtZero: true, ticks: {{ font: {{ size: 11 }} }} }}
    }}
  }}
}});

// ── Applicant lookup ──────────────────────────────────────────────────────────
async function lookupApplicant() {{
  const id  = document.getElementById('applicant-input').value.trim();
  const div = document.getElementById('applicant-result');
  if (!id) return;

  div.innerHTML = '<p style="color:#64748b;font-size:13px">Searching…</p>';

  try {{
    const res  = await fetch(`/dashboard/api/applicant/${{encodeURIComponent(id)}}`);
    const data = await res.json();

    if (!data.length) {{
      div.innerHTML = '<p style="color:#94a3b8;font-size:13px">No records found for this applicant.</p>';
      return;
    }}

    let html = `<p style="font-size:12px;color:#64748b;margin-bottom:10px">${{data.length}} record(s) found</p>
    <table style="font-size:12px">
      <thead><tr>
        <th>Date</th><th>File</th><th>Type</th><th>Risk</th><th>Score</th><th>Flags</th>
      </tr></thead><tbody>`;

    for (const r of data) {{
      const flags = Array.isArray(r.flags) ? r.flags.length : 0;
      const color = r.risk_level === 'HIGH' ? '#ef4444' : r.risk_level === 'MEDIUM' ? '#f59e0b' : '#22c55e';
      html += `<tr>
        <td>${{(r.screened_at||'').substring(0,19).replace('T',' ')}}</td>
        <td>${{r.file_name||'—'}}</td>
        <td>${{(r.doc_type||'').toUpperCase()}}</td>
        <td style="font-weight:700;color:${{color}}">${{r.risk_level||'—'}}</td>
        <td>${{r.risk_score||'—'}}</td>
        <td>${{flags}} flag(s)</td>
      </tr>`;
    }}
    html += '</tbody></table>';
    div.innerHTML = html;
  }} catch(e) {{
    div.innerHTML = '<p style="color:#ef4444;font-size:13px">Error fetching data.</p>';
  }}
}}

document.getElementById('applicant-input').addEventListener('keydown', e => {{
  if (e.key === 'Enter') lookupApplicant();
}});
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "running", "service": "dashboard"}
