"""
telegram_alerts.py — Writes alerts to telegram_queue table.
The telegram_bridge.py service running on the HOST server
reads the queue and sends messages (host has internet access).
"""
import os, json, logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log       = logging.getLogger("telegram_alerts")
ENABLED   = os.getenv("TELEGRAM_ALERTS_ENABLED","false").lower() == "true"
BANK_NAME = os.getenv("BANK_NAME","Bank")
SERVER_IP = os.getenv("SERVER_IP","172.16.26.48")
DATABASE_URL = os.getenv("DATABASE_URL","")

_engine = None
def _get_engine():
    global _engine
    if _engine is None and DATABASE_URL:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine

def _queue(message: str) -> dict:
    """Write message to telegram_queue table for bridge to send."""
    if not ENABLED:
        log.info(f"Telegram disabled: {message[:60]}")
        return {"ok": False, "reason": "disabled"}
    try:
        eng = _get_engine()
        if not eng:
            return {"ok": False, "error": "no_database"}
        with eng.connect() as db:
            db.execute(text("""
                INSERT INTO telegram_queue (message)
                VALUES (:msg)
            """), {"msg": message})
            db.commit()
        log.info("Alert queued for Telegram bridge")
        return {"ok": True, "queued": True}
    except Exception as e:
        log.error(f"Queue failed: {e}")
        return {"ok": False, "error": str(e)}

def _link(path): return f"http://{SERVER_IP}:3000{path}"
def _fmt(amount):
    try: return f"${float(amount):,.0f}"
    except: return str(amount) if amount else "—"
def _risk_emoji(level): return {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(level,"⚪")

def alert_new_application(loan_ref, applicant_name, loan_type,
                           loan_amount, source="applicant-portal",
                           applicant_id="") -> dict:
    src = "👤 Customer" if source=="applicant-portal" else f"👔 {source}"
    msg = (
        f"📋 <b>New Loan Application</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 Ref:    <code>{loan_ref}</code>\n"
        f"👤 Name:   {applicant_name or '—'}\n"
        f"💰 Amount: {_fmt(loan_amount)}\n"
        f"📄 Type:   {loan_type or '—'}\n"
        f"📥 From:   {src}\n"
        f"🕐 Time:   {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔗 <a href='{_link(f'/loan/case/{loan_ref}')}'>View Case</a>"
    )
    return _queue(msg)

def alert_kyc_complete(loan_ref, applicant_name, risk_level,
                       risk_score, action, flags, loan_amount=None) -> dict:
    if risk_level == "LOW":
        return {"ok": True, "reason": "LOW — silent"}
    emoji     = _risk_emoji(risk_level)
    act_emoji = {"REJECT":"🚫","REVIEW":"⚠️","PASS":"✅"}.get(action,"❓")
    top       = (flags or [])[:5]
    flags_txt = "\n".join(f"  ⚠ <code>{f}</code>" for f in top)
    if len(flags or []) > 5: flags_txt += f"\n  +{len(flags)-5} more"
    msg = (
        f"{emoji} <b>KYC Alert — {risk_level}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 Ref:    <code>{loan_ref}</code>\n"
        f"👤 Name:   {applicant_name or '—'}\n"
        f"💰 Amount: {_fmt(loan_amount)}\n"
        f"📊 Score:  {risk_score}\n"
        f"{act_emoji} Action: {action}\n"
        f"🚩 Flags:\n{flags_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔗 <a href='{_link(f'/loan/case/{loan_ref}')}'>Review</a>"
    )
    return _queue(msg)

def alert_decision(loan_ref, applicant_name, decision, officer,
                   notes="", loan_amount=None, interest_rate=None) -> dict:
    emoji = {"approved":"✅","rejected":"❌","review":"🔄"}.get(decision,"📋")
    label = {"approved":"APPROVED","rejected":"REJECTED",
             "review":"SENT FOR REVIEW"}.get(decision, decision.upper())
    msg = (
        f"{emoji} <b>Decision: {label}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 Ref:      <code>{loan_ref}</code>\n"
        f"👤 Name:     {applicant_name or '—'}\n"
        f"💰 Amount:   {_fmt(loan_amount)}\n"
        f"👔 Officer:  {officer}\n"
        f"🕐 Time:     {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )
    if notes: msg += f"📝 Notes: {notes[:150]}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n🔗 <a href='{_link(f'/loan/case/{loan_ref}')}'>View</a>"
    return _queue(msg)

def alert_high_risk_kyc(file_name, applicant_id, risk_score,
                        flags, doc_type="") -> dict:
    top       = (flags or [])[:6]
    flags_txt = "\n".join(f"  🚩 <code>{f}</code>" for f in top)
    msg = (
        f"🔴 <b>HIGH Risk Document</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📁 File:  <code>{str(file_name)[:45]}</code>\n"
        f"📄 Type:  {doc_type or '—'}\n"
        f"📊 Score: {risk_score}\n"
        f"🚩 Flags:\n{flags_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return _queue(msg)

def alert_fraud_network(applicant_id, connections, link_types, linked_to) -> dict:
    msg = (
        f"🕸 <b>Fraud Network Alert</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 Applicant:   <code>{applicant_id}</code>\n"
        f"🔗 Connections: {connections}\n"
        f"📎 Shared via:  {', '.join((link_types or [])[:4])}\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return _queue(msg)

def alert_anomaly(file_name, applicant_id, score, severity) -> dict:
    if severity not in ("high","medium"):
        return {"ok": True, "reason": "low"}
    emoji = "🔴" if severity=="high" else "🟡"
    msg = (
        f"{emoji} <b>Anomaly — {severity.upper()}</b>\n"
        f"📁 <code>{str(file_name)[:45]}</code>\n"
        f"📊 Score: {score:.4f}\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return _queue(msg)

def alert_daily_summary() -> dict:
    try:
        eng = _get_engine()
        with eng.connect() as db:
            row = db.execute(text("""
                SELECT COUNT(*) total,
                    COUNT(*) FILTER (WHERE risk_level='HIGH')   high,
                    COUNT(*) FILTER (WHERE risk_level='MEDIUM') medium,
                    COUNT(*) FILTER (WHERE risk_level='LOW')    low,
                    COUNT(DISTINCT applicant_id) apps
                FROM screening_logs
                WHERE screened_at >= NOW()-INTERVAL '24 hours'
            """)).fetchone()
        msg = (
            f"📊 <b>Daily Summary — {datetime.utcnow().strftime('%Y-%m-%d')}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 Total:  {row[0] or 0}\n"
            f"🔴 HIGH:   {row[1] or 0}\n"
            f"🟡 MEDIUM: {row[2] or 0}\n"
            f"🟢 LOW:    {row[3] or 0}\n"
            f"👥 Apps:   {row[4] or 0}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 <a href='{_link('/dashboard/')}'>Dashboard</a>"
        )
        return _queue(msg)
    except Exception as e:
        return {"ok": False, "error": str(e)}

def test_alert() -> dict:
    msg = (
        f"✅ <b>Telegram Bridge Active</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏦 {BANK_NAME} KYC System\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 New applications\n"
        f"🔴 HIGH risk alerts\n"
        f"✅ Loan decisions\n"
        f"📊 Daily summary"
    )
    return _queue(msg)

if __name__ == "__main__":
    print("Queuing test alert...")
    result = test_alert()
    print("Result:", json.dumps(result, indent=2))
    if result.get("ok"):
        print("✅ Queued — bridge will send it within 5 seconds")
    else:
        print(f"❌ Failed: {result.get('error')}")
