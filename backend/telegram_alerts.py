"""
telegram_alerts.py — Telegram Bot Alert System
Sends real-time alerts to Telegram when:
  - New loan application submitted by customer
  - KYC screening completes with HIGH risk
  - Loan officer makes a decision (approve/reject)
  - Fraud network connection detected
  - Anomaly detected in document

Setup:
  1. Message @BotFather on Telegram → /newbot → copy token
  2. Message @userinfobot → copy your chat ID
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=7123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TELEGRAM_CHAT_ID=-1001234567890
       TELEGRAM_ALERTS_ENABLED=true
"""

import os
import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("telegram_alerts")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLED   = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"
BANK_NAME = os.getenv("BANK_NAME", "Bank")
SERVER_IP = os.getenv("SERVER_IP", "172.16.26.48")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


# ── Core sender ───────────────────────────────────────────────────────────────

def _send(message: str, chat_id: str = None, parse_mode: str = "HTML") -> dict:
    """Send a message to Telegram."""
    if not ENABLED:
        log.info(f"Telegram disabled — would send: {message[:80]}")
        return {"ok": False, "reason": "TELEGRAM_ALERTS_ENABLED=false"}

    if not BOT_TOKEN or not (chat_id or CHAT_ID):
        log.error("Telegram credentials not configured")
        return {"ok": False, "error": "credentials_missing"}

    target = chat_id or CHAT_ID

    try:
        data = json.dumps({
            "chat_id":    target,
            "text":       message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{TELEGRAM_API}/sendMessage",
            data    = data,
            method  = "POST",
            headers = {"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                log.info(f"Telegram alert sent — message_id={result['result']['message_id']}")
            return result

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error(f"Telegram HTTP error {e.code}: {body}")
        return {"ok": False, "error": f"http_{e.code}", "detail": body}
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return {"ok": False, "error": str(e)}


def _risk_emoji(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")


def _fmt_amount(amount) -> str:
    try:
        return f"${float(amount):,.0f}"
    except (TypeError, ValueError):
        return str(amount) if amount else "—"


def _portal_link(path: str) -> str:
    return f"http://{SERVER_IP}:3000{path}"


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def alert_new_application(
    loan_ref:       str,
    applicant_name: str,
    loan_type:      str,
    loan_amount,
    source:         str = "applicant-portal",
    applicant_id:   str = "",
) -> dict:
    """
    Fire when a new loan application is submitted.
    Called from applicant_portal.py and loan_portal.py.
    """
    source_label = "👤 Customer (Self-Service)" if source == "applicant-portal" \
                   else "👔 Loan Officer"
    msg = (
        f"📋 <b>New Loan Application</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 Reference:  <code>{loan_ref}</code>\n"
        f"👤 Applicant:  {applicant_name or '—'}\n"
        f"💰 Amount:     {_fmt_amount(loan_amount)}\n"
        f"📄 Type:       {loan_type or '—'}\n"
        f"📥 Submitted:  {source_label}\n"
        f"🕐 Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔗 <a href='{_portal_link(f'/loan/case/{loan_ref}')}'>View Case</a>"
    )
    return _send(msg)


def alert_kyc_complete(
    loan_ref:     str,
    applicant_name: str,
    risk_level:   str,
    risk_score:   int,
    action:       str,
    flags:        list,
    loan_amount = None,
) -> dict:
    """
    Fire when KYC screening completes.
    Only alerts for HIGH and MEDIUM risk — LOW is silent.
    """
    if risk_level == "LOW":
        return {"ok": True, "reason": "LOW risk — no alert needed"}

    emoji   = _risk_emoji(risk_level)
    top_flags = flags[:5] if flags else []
    flags_text = "\n".join(f"  ⚠ <code>{f}</code>" for f in top_flags)
    if len(flags) > 5:
        flags_text += f"\n  ... and {len(flags)-5} more"

    action_emoji = {"REJECT": "🚫", "REVIEW": "⚠️", "PASS": "✅"}.get(action, "❓")

    msg = (
        f"{emoji} <b>KYC Alert — {risk_level} Risk</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 Reference:  <code>{loan_ref}</code>\n"
        f"👤 Applicant:  {applicant_name or '—'}\n"
        f"💰 Amount:     {_fmt_amount(loan_amount)}\n"
        f"📊 Risk Score: {risk_score}\n"
        f"{action_emoji} Action:     {action}\n"
    )
    if flags_text:
        msg += f"🚩 Flags:\n{flags_text}\n"
    msg += (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔗 <a href='{_portal_link(f'/loan/case/{loan_ref}')}'>Review Now</a>"
    )
    return _send(msg)


def alert_decision(
    loan_ref:       str,
    applicant_name: str,
    decision:       str,
    officer:        str,
    notes:          str = "",
    loan_amount     = None,
    interest_rate   = None,
) -> dict:
    """Fire when a loan officer makes a decision (approve/reject)."""
    emoji = {"approved": "✅", "rejected": "❌", "review": "🔄"}.get(decision, "📋")
    label = {"approved": "APPROVED", "rejected": "REJECTED",
             "review": "SENT FOR REVIEW"}.get(decision, decision.upper())

    msg = (
        f"{emoji} <b>Loan Decision: {label}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 Reference:  <code>{loan_ref}</code>\n"
        f"👤 Applicant:  {applicant_name or '—'}\n"
        f"💰 Amount:     {_fmt_amount(loan_amount)}\n"
    )
    if interest_rate and decision == "approved":
        msg += f"📈 Rate:       {interest_rate}% p.a.\n"
    msg += (
        f"👔 Officer:    {officer}\n"
        f"🕐 Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )
    if notes:
        msg += f"📝 Notes:      {notes[:200]}\n"
    msg += (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔗 <a href='{_portal_link(f'/loan/case/{loan_ref}')}'>View Case</a>"
    )
    return _send(msg)


def alert_high_risk_kyc(
    file_name:    str,
    applicant_id: str,
    risk_score:   int,
    flags:        list,
    doc_type:     str = "",
) -> dict:
    """
    Fire when KYC screening (not loan) detects HIGH risk document.
    Called from database.py alert trigger.
    """
    top_flags  = flags[:6] if flags else []
    flags_text = "\n".join(f"  🚩 <code>{f}</code>" for f in top_flags)
    if len(flags) > 6:
        flags_text += f"\n  ... +{len(flags)-6} more"

    msg = (
        f"🔴 <b>HIGH Risk Document Detected</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📁 File:       <code>{file_name[:50]}</code>\n"
        f"🆔 Applicant:  {applicant_id or '—'}\n"
        f"📄 Type:       {doc_type or '—'}\n"
        f"📊 Score:      {risk_score}\n"
        f"🚩 Flags:\n{flags_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"🔗 <a href='{_portal_link('/portal/cases')}'>View in KYC Portal</a>"
    )
    return _send(msg)


def alert_fraud_network(
    applicant_id:  str,
    connections:   int,
    link_types:    list,
    linked_to:     list,
) -> dict:
    """Fire when fraud network connections are detected."""
    links_text = ", ".join(link_types[:4])
    linked_ids = ", ".join(f"<code>{l}</code>" for l in linked_to[:3])

    msg = (
        f"🕸 <b>Fraud Network Alert</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 Applicant:     <code>{applicant_id}</code>\n"
        f"🔗 Connections:   {connections} other applicants\n"
        f"📎 Shared via:    {links_text}\n"
        f"👥 Linked to:     {linked_ids}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"🔗 <a href='{_portal_link('/portal/cases')}'>Investigate</a>"
    )
    return _send(msg)


def alert_anomaly(
    file_name:    str,
    applicant_id: str,
    score:        float,
    severity:     str,
) -> dict:
    """Fire when anomaly detector flags a document."""
    if severity not in ("high", "medium"):
        return {"ok": True, "reason": "low severity — no alert"}

    emoji = "🔴" if severity == "high" else "🟡"
    msg = (
        f"{emoji} <b>Anomaly Detected — {severity.upper()}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📁 File:       <code>{file_name[:50]}</code>\n"
        f"🆔 Applicant:  {applicant_id or '—'}\n"
        f"📊 Score:      {score:.4f}\n"
        f"ℹ️  This document looks statistically different\n"
        f"   from your genuine document baseline.\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"🔗 <a href='{_portal_link('/portal/cases')}'>Review</a>"
    )
    return _send(msg)


def alert_daily_summary() -> dict:
    """
    Send a daily summary of screening activity.
    Run via cron: 0 8 * * * docker exec backend-api-1 python -c
    "from telegram_alerts import alert_daily_summary; alert_daily_summary()"
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL = os.getenv("DATABASE_URL", "")
    if not DATABASE_URL:
        return {"ok": False, "error": "no database url"}

    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        db     = sessionmaker(bind=engine)()
        rows   = db.execute(text("""
            SELECT
                COUNT(*)                                        AS total,
                COUNT(*) FILTER (WHERE risk_level='HIGH')      AS high,
                COUNT(*) FILTER (WHERE risk_level='MEDIUM')    AS medium,
                COUNT(*) FILTER (WHERE risk_level='LOW')       AS low,
                COUNT(DISTINCT applicant_id)                   AS unique_apps
            FROM screening_logs
            WHERE screened_at >= NOW() - INTERVAL '24 hours'
        """)).fetchone()
        db.close()

        if not rows:
            return {"ok": False, "error": "no data"}

        total, high, medium, low, unique = rows
        msg = (
            f"📊 <b>Daily Screening Summary</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Date:       {datetime.utcnow().strftime('%Y-%m-%d')}\n"
            f"📋 Total:      {total or 0} screenings\n"
            f"🔴 HIGH:       {high or 0}\n"
            f"🟡 MEDIUM:     {medium or 0}\n"
            f"🟢 LOW:        {low or 0}\n"
            f"👥 Applicants: {unique or 0} unique\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 <a href='{_portal_link('/dashboard/')}'>View Dashboard</a>"
        )
        return _send(msg)

    except Exception as e:
        log.error(f"Daily summary failed: {e}")
        return {"ok": False, "error": str(e)}


# ── Test ──────────────────────────────────────────────────────────────────────

def test_alert() -> dict:
    """Send a test alert to verify configuration."""
    msg = (
        f"✅ <b>Telegram Alerts Active</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏦 System:  {BANK_NAME} KYC\n"
        f"🕐 Time:    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Alerts configured for:\n"
        f"  📋 New applications\n"
        f"  🔴 HIGH risk KYC results\n"
        f"  ✅ Loan decisions\n"
        f"  🕸 Fraud network detection\n"
        f"  ⚠️  Anomaly detection\n"
        f"  📊 Daily summary"
    )
    return _send(msg)


if __name__ == "__main__":
    import sys
    print("Testing Telegram alert...")
    result = test_alert()
    print("Result:", json.dumps(result, indent=2))
    if result.get("ok"):
        print("\n✅ Success — check your Telegram!")
    else:
        print(f"\n❌ Failed: {result.get('error','unknown')}")
        if "credentials_missing" in str(result):
            print("\nSet in .env:")
            print("  TELEGRAM_BOT_TOKEN=your_token")
            print("  TELEGRAM_CHAT_ID=your_chat_id")
            print("  TELEGRAM_ALERTS_ENABLED=true")
