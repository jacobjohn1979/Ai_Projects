"""
webhook_receiver.py — Simple webhook receiver for ID card screening callbacks
Run this as a separate service on port 8001 (internal) or expose via Nginx.
Receives POST callbacks from the fraud detection API and stores/displays results.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("webhook_receiver")

# ── Simple file-based store (no extra DB needed) ──────────────────────────────
WEBHOOK_LOG_DIR = Path(os.getenv("WEBHOOK_LOG_DIR", "webhook_logs"))
WEBHOOK_LOG_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")   # optional shared secret

app = FastAPI(
    title="Fraud Detection Webhook Receiver",
    description="Receives and stores ID card screening callbacks",
    version="1.0.0",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_event(payload: dict) -> str:
    """Save webhook payload to a JSON file, return the file path."""
    task_id    = payload.get("task_id", "unknown")
    timestamp  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename   = f"{timestamp}_{task_id}.json"
    file_path  = WEBHOOK_LOG_DIR / filename

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return str(file_path)


def _load_events(limit: int = 50) -> list:
    """Load recent webhook events from disk, newest first."""
    files = sorted(WEBHOOK_LOG_DIR.glob("*.json"), reverse=True)[:limit]
    events = []
    for f in files:
        try:
            with open(f) as fp:
                events.append(json.load(fp))
        except Exception:
            pass
    return events


def _risk_color(level: str) -> str:
    return {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(level, "#6b7280")


def _action_badge(action: str) -> str:
    colors = {"REJECT": "#ef4444", "REVIEW": "#f59e0b", "PASS": "#22c55e"}
    color  = colors.get(action, "#6b7280")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:4px;font-weight:bold;font-size:12px">{action}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBHOOK RECEIVER ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/kyc", summary="Receive ID card screening callback")
async def receive_webhook(request: Request):
    """
    Callback URL to provide to /screen-id-card as callback_url:
      http://your-server-ip:8088/webhook/kyc
    """
    # ── Optional secret validation ────────────────────────────────────────────
    if WEBHOOK_SECRET:
        provided = request.headers.get("X-Webhook-Secret", "")
        if provided != WEBHOOK_SECRET:
            raise HTTPException(403, "Invalid webhook secret")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON payload")

    # ── Log to console ────────────────────────────────────────────────────────
    event       = payload.get("event", "unknown")
    task_id     = payload.get("task_id", "unknown")
    applicant   = payload.get("applicant_id", "unknown")
    risk        = payload.get("risk", {})
    risk_level  = risk.get("level", "UNKNOWN")
    action      = risk.get("action", "UNKNOWN")

    log.info(f"[WEBHOOK] event={event} task={task_id} applicant={applicant} "
             f"risk={risk_level} action={action}")

    # ── Save to disk ──────────────────────────────────────────────────────────
    saved_path = _save_event(payload)
    log.info(f"[WEBHOOK] Saved to {saved_path}")

    # ── Return 200 immediately (fraud API checks for success) ─────────────────
    return JSONResponse({
        "received":   True,
        "task_id":    task_id,
        "event":      event,
        "saved_path": saved_path,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD — view received webhooks
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    """Simple HTML dashboard to view received webhook events."""
    events = _load_events(50)

    rows = ""
    for ev in events:
        risk       = ev.get("risk", {})
        level      = risk.get("level", "—")
        action     = risk.get("action", "—")
        score      = risk.get("score", "—")
        task_id    = ev.get("task_id", "—")
        applicant  = ev.get("applicant_id") or "—"
        filename   = ev.get("file_name", "—")
        screened   = ev.get("screened_at", "—")
        event_type = ev.get("event", "—")
        flags      = ev.get("flags", [])
        flag_count = len(flags)
        color      = _risk_color(level)

        flag_html = ""
        if flags:
            flag_html = "".join(
                f'<span style="background:#f3f4f6;border:1px solid #e5e7eb;'
                f'border-radius:3px;padding:2px 6px;font-size:11px;margin:2px;display:inline-block">'
                f'{f}</span>'
                for f in flags[:8]
            )
            if len(flags) > 8:
                flag_html += f'<span style="color:#6b7280;font-size:11px"> +{len(flags)-8} more</span>'

        rows += f"""
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:12px 8px;font-size:12px;color:#6b7280">{screened[:19].replace("T"," ")}</td>
          <td style="padding:12px 8px;font-size:12px">{filename}</td>
          <td style="padding:12px 8px;font-size:12px;color:#6b7280">{applicant}</td>
          <td style="padding:12px 8px;text-align:center">
            <span style="color:{color};font-weight:bold">{level}</span>
          </td>
          <td style="padding:12px 8px;text-align:center">{score}</td>
          <td style="padding:12px 8px;text-align:center">{_action_badge(action)}</td>
          <td style="padding:12px 8px;font-size:11px">{flag_count} flag{"s" if flag_count != 1 else ""}<br>{flag_html}</td>
          <td style="padding:12px 8px;font-size:10px;color:#9ca3af;word-break:break-all">{task_id[:18]}…</td>
        </tr>
        """

    total   = len(events)
    high    = sum(1 for e in events if e.get("risk", {}).get("level") == "HIGH")
    medium  = sum(1 for e in events if e.get("risk", {}).get("level") == "MEDIUM")
    low     = sum(1 for e in events if e.get("risk", {}).get("level") == "LOW")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Fraud Detection — Webhook Dashboard</title>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f9fafb; color: #111827; }}
    .header {{ background: #1e293b; color: white; padding: 20px 32px;
               display: flex; align-items: center; justify-content: space-between; }}
    .header h1 {{ font-size: 18px; font-weight: 600; }}
    .header p  {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .badge {{ background: #334155; border-radius: 4px; padding: 4px 10px;
              font-size: 12px; color: #e2e8f0; }}
    .stats {{ display: flex; gap: 16px; padding: 20px 32px; }}
    .stat {{ background: white; border-radius: 8px; padding: 16px 24px;
             border: 1px solid #e5e7eb; flex: 1; text-align: center; }}
    .stat .num {{ font-size: 28px; font-weight: 700; }}
    .stat .lbl {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
    .table-wrap {{ padding: 0 32px 32px; }}
    table {{ width: 100%; background: white; border-radius: 8px;
             border: 1px solid #e5e7eb; border-collapse: collapse; }}
    thead {{ background: #f8fafc; }}
    th {{ padding: 12px 8px; text-align: left; font-size: 12px;
          font-weight: 600; color: #374151; border-bottom: 1px solid #e5e7eb; }}
    tr:hover {{ background: #f8fafc; }}
    .refresh {{ font-size: 11px; color: #94a3b8; }}
    .webhook-url {{ background: #1e293b; color: #a5f3fc; padding: 12px 32px;
                    font-family: monospace; font-size: 13px; }}
  </style>
</head>
<body>

<div class="header">
  <div>
    <h1>🔍 Fraud Detection — Webhook Dashboard</h1>
    <p>Receiving ID card screening callbacks · Auto-refreshes every 10 seconds</p>
  </div>
  <div>
    <span class="badge">Last {total} events</span>
    &nbsp;
    <span class="refresh">⟳ Auto-refresh ON</span>
  </div>
</div>

<div class="webhook-url">
  📡 Callback URL: &nbsp;
  <strong>http://YOUR_SERVER_IP:8088/webhook/kyc</strong>
  &nbsp;— use this as <code>callback_url</code> when submitting ID cards
</div>

<div class="stats">
  <div class="stat">
    <div class="num">{total}</div>
    <div class="lbl">Total Screened</div>
  </div>
  <div class="stat">
    <div class="num" style="color:#ef4444">{high}</div>
    <div class="lbl">HIGH Risk</div>
  </div>
  <div class="stat">
    <div class="num" style="color:#f59e0b">{medium}</div>
    <div class="lbl">MEDIUM Risk</div>
  </div>
  <div class="stat">
    <div class="num" style="color:#22c55e">{low}</div>
    <div class="lbl">LOW Risk</div>
  </div>
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Screened At</th>
        <th>File</th>
        <th>Applicant ID</th>
        <th style="text-align:center">Risk</th>
        <th style="text-align:center">Score</th>
        <th style="text-align:center">Action</th>
        <th>Flags</th>
        <th>Task ID</th>
      </tr>
    </thead>
    <tbody>
      {"".join([rows]) if rows else
       '<tr><td colspan="8" style="text-align:center;padding:40px;color:#9ca3af">No webhook events received yet.<br>Submit an ID card to <code>/screen-id-card</code> with a callback_url.</td></tr>'}
    </tbody>
  </table>
</div>

</body>
</html>"""

    return HTMLResponse(html)


@app.get("/events", summary="Get recent webhook events as JSON")
def get_events(limit: int = 50):
    """Returns recent webhook events as JSON — useful for integrating into your own system."""
    return JSONResponse({"events": _load_events(limit), "count": len(_load_events(limit))})


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "running", "service": "webhook-receiver"}
