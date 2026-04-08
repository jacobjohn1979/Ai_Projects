"""
alerts.py — Email alert system
Sends email when HIGH risk document detected, plus daily digest.
"""
import os
import json
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("fraud_detect.alerts")

SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM    = os.getenv("ALERT_FROM", SMTP_USER)
ALERT_TO      = os.getenv("ALERT_TO", "")      # comma-separated emails
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "false").lower() == "true"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

RISK_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


def _send_email(to_list: list, subject: str, html_body: str):
    if not ALERTS_ENABLED:
        log.info(f"Alerts disabled — would send: {subject}")
        return False
    if not SMTP_USER or not SMTP_PASSWORD:
        log.warning("SMTP credentials not configured")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = ALERT_FROM
        msg["To"]      = ", ".join(to_list)
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(ALERT_FROM, to_list, msg.as_string())
        log.info(f"Alert sent to {to_list}: {subject}")
        return True
    except Exception as e:
        log.error(f"Email send failed: {e}")
        return False


def _get_recipients() -> list:
    if not ALERT_TO:
        return []
    return [e.strip() for e in ALERT_TO.split(",") if e.strip()]


def send_high_risk_alert(result: dict, filename: str, applicant_id: str = None):
    """Send immediate alert when HIGH risk document is detected."""
    recipients = _get_recipients()
    if not recipients:
        return

    risk    = result.get("risk", {})
    level   = risk.get("level", "HIGH")
    score   = risk.get("score", 0)
    action  = risk.get("action", "REJECT")
    flags   = result.get("flags", [])
    screened = result.get("screened_at", datetime.utcnow().isoformat())[:19]

    flag_rows = "".join([
        '<tr><td style="padding:4px 12px;font-family:monospace;font-size:12px;'
        'background:#fef2f2;border-radius:3px;margin:2px">' + f + '</td></tr>'
        for f in flags[:10]
    ])

    html = f"""
    <div style="font-family:-apple-system,sans-serif;max-width:600px;margin:0 auto">
      <div style="background:#ef4444;color:#fff;padding:20px 24px;border-radius:8px 8px 0 0">
        <h2 style="margin:0;font-size:18px">🔴 HIGH RISK Document Detected</h2>
        <p style="margin:6px 0 0;opacity:.9;font-size:13px">Immediate review required</p>
      </div>
      <div style="background:#fff;border:1px solid #e2e8f0;padding:24px;border-radius:0 0 8px 8px">
        <table style="width:100%;border-collapse:collapse;margin-bottom:16px">
          <tr><td style="color:#64748b;padding:6px 0;width:40%">File Name</td>
              <td style="font-weight:600">{filename}</td></tr>
          <tr><td style="color:#64748b;padding:6px 0">Applicant ID</td>
              <td>{applicant_id or "—"}</td></tr>
          <tr><td style="color:#64748b;padding:6px 0">Risk Level</td>
              <td style="color:#ef4444;font-weight:700">{level}</td></tr>
          <tr><td style="color:#64748b;padding:6px 0">Risk Score</td>
              <td style="font-weight:600">{score}</td></tr>
          <tr><td style="color:#64748b;padding:6px 0">Action</td>
              <td style="color:#ef4444;font-weight:700">{action}</td></tr>
          <tr><td style="color:#64748b;padding:6px 0">Screened At</td>
              <td>{screened}</td></tr>
          <tr><td style="color:#64748b;padding:6px 0">Flags ({len(flags)})</td>
              <td></td></tr>
          {flag_rows}
        </table>
        <a href="http://{os.getenv('SERVER_IP','localhost')}:3000/portal/cases"
           style="background:#ef4444;color:#fff;padding:10px 20px;border-radius:6px;
                  text-decoration:none;font-size:13px;font-weight:500">
          Review in Portal →
        </a>
        <p style="color:#94a3b8;font-size:11px;margin-top:16px">
          This is an automated alert from the KYC Fraud Detection System.
        </p>
      </div>
    </div>"""

    _send_email(recipients, f"[HIGH RISK] {filename} — Score: {score}", html)


def send_daily_digest():
    """Send daily summary of screening activity."""
    recipients = _get_recipients()
    if not recipients:
        return

    since = datetime.utcnow() - timedelta(hours=24)
    db    = SessionLocal()
    try:
        stats = db.execute(text("""
            SELECT COUNT(*) total,
                   COUNT(*) FILTER (WHERE risk_level='HIGH')   high,
                   COUNT(*) FILTER (WHERE risk_level='MEDIUM') medium,
                   COUNT(*) FILTER (WHERE risk_level='LOW')    low,
                   COUNT(*) FILTER (WHERE doc_type='id_card')  id_cards,
                   COUNT(*) FILTER (WHERE doc_type='pdf')      pdfs
            FROM screening_logs WHERE screened_at >= :s
        """), {"s": since}).fetchone()

        if not stats or stats.total == 0:
            log.info("No activity in last 24h — skipping digest")
            return

        s = dict(stats._mapping)

        high_cases = db.execute(text("""
            SELECT file_name, applicant_id, risk_score, id_number
            FROM screening_logs
            WHERE risk_level='HIGH' AND screened_at >= :s
            ORDER BY risk_score DESC LIMIT 5
        """), {"s": since}).fetchall()

        high_rows = "".join([
            '<tr style="border-bottom:1px solid #f1f5f9">'
            '<td style="padding:8px 12px">' + str(r.file_name or "—")[:30] + '</td>'
            '<td style="padding:8px 12px">' + str(r.applicant_id or "—") + '</td>'
            '<td style="padding:8px 12px;font-weight:700;color:#ef4444">' + str(r.risk_score) + '</td>'
            '</tr>'
            for r in high_cases
        ])

        html = f"""
        <div style="font-family:-apple-system,sans-serif;max-width:600px;margin:0 auto">
          <div style="background:#0f172a;color:#fff;padding:20px 24px;border-radius:8px 8px 0 0">
            <h2 style="margin:0;font-size:18px">📊 Daily KYC Screening Report</h2>
            <p style="margin:6px 0 0;opacity:.7;font-size:13px">
              {datetime.utcnow().strftime("%d %B %Y")} — Last 24 hours
            </p>
          </div>
          <div style="background:#fff;border:1px solid #e2e8f0;padding:24px;border-radius:0 0 8px 8px">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px">
              <div style="text-align:center;padding:16px;background:#f8fafc;border-radius:8px">
                <div style="font-size:28px;font-weight:700">{s["total"]}</div>
                <div style="font-size:12px;color:#64748b">Total Screened</div>
              </div>
              <div style="text-align:center;padding:16px;background:#fef2f2;border-radius:8px">
                <div style="font-size:28px;font-weight:700;color:#ef4444">{s["high"]}</div>
                <div style="font-size:12px;color:#64748b">HIGH Risk</div>
              </div>
              <div style="text-align:center;padding:16px;background:#fffbeb;border-radius:8px">
                <div style="font-size:28px;font-weight:700;color:#f59e0b">{s["medium"]}</div>
                <div style="font-size:12px;color:#64748b">MEDIUM Risk</div>
              </div>
            </div>
            {"<h3 style='font-size:13px;font-weight:600;color:#64748b;margin-bottom:8px'>HIGH Risk Cases Today</h3><table style='width:100%;border-collapse:collapse;font-size:13px'><thead><tr style='background:#f8fafc'><th style='padding:8px 12px;text-align:left'>File</th><th style='padding:8px 12px;text-align:left'>Applicant</th><th style='padding:8px 12px;text-align:left'>Score</th></tr></thead><tbody>" + high_rows + "</tbody></table>" if high_cases else "<p style='color:#22c55e;font-weight:600'>✅ No HIGH risk cases today</p>"}
            <div style="margin-top:20px">
              <a href="http://{os.getenv('SERVER_IP','localhost')}:3000/portal/"
                 style="background:#3b82f6;color:#fff;padding:10px 20px;border-radius:6px;
                        text-decoration:none;font-size:13px;font-weight:500">
                Open Portal →
              </a>
            </div>
          </div>
        </div>"""

        _send_email(recipients,
                    f"KYC Daily Report — {s['total']} screened, {s['high']} HIGH risk",
                    html)
    finally:
        db.close()


def test_email_config() -> dict:
    """Test email configuration. Returns status dict."""
    recipients = _get_recipients()
    return {
        "alerts_enabled": ALERTS_ENABLED,
        "smtp_host":      SMTP_HOST,
        "smtp_port":      SMTP_PORT,
        "smtp_user":      SMTP_USER or "not set",
        "recipients":     recipients,
        "configured":     bool(SMTP_USER and SMTP_PASSWORD and recipients),
    }
