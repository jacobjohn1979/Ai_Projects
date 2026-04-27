"""
applicant_notifications.py — Multi-channel Applicant Notification System
Sends real-time updates to loan applicants via:
  - Telegram (bot message)
  - SMS (Twilio)
  - Email (SMTP/Gmail)

Triggered on every status change:
  submitted  → Application received
  screening  → KYC screening started
  review     → Under review by officer
  approved   → Congratulations, approved
  rejected   → Not approved

Setup:
  All config via .env — see bottom of file for required variables.
  The Telegram bot for applicants is the SAME bot as staff alerts
  but sends to the applicant's personal Telegram chat ID.

Applicant Telegram onboarding:
  1. Applicant messages @your_bot with /start
  2. Bot replies asking for their loan reference
  3. Applicant sends their ref (e.g. AP2604173144)
  4. Bot stores their Telegram chat_id linked to that loan ref
  5. All future updates go directly to their Telegram
"""

import os
import json
import logging
import smtplib
import subprocess
import tempfile
import re
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("applicant_notifications")

# ── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
SERVER_IP    = os.getenv("SERVER_IP", "172.16.26.48")
BANK_NAME    = os.getenv("BANK_NAME", "Bank")
BANK_PHONE   = os.getenv("BANK_PHONE", "")
BANK_EMAIL   = os.getenv("BANK_EMAIL", os.getenv("SMTP_USER", ""))
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASSWORD", "")
SMS_ENABLED  = os.getenv("SMS_ENABLED", "false").lower() == "true"
TG_ENABLED   = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"
EMAIL_ENABLED= os.getenv("ALERTS_ENABLED", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "")

_engine = None
def _db():
    global _engine
    if not _engine:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return sessionmaker(bind=_engine)()


# ── Message templates (EN + KM) ───────────────────────────────────────────────

TEMPLATES = {
    "submitted": {
        "subject_en": "Application Received — {ref}",
        "subject_km": "ពាក្យសុំត្រូវបានទទួល — {ref}",
        "en": (
            "Dear {name},\n\n"
            "Your loan application {ref} has been received successfully.\n\n"
            "📋 Loan Type:   {loan_type}\n"
            "💰 Amount:      ${amount:,.0f}\n"
            "📅 Submitted:   {date}\n\n"
            "We will begin reviewing your documents shortly.\n"
            "Track your application: http://{ip}:3000/apply/\n\n"
            "Questions? Contact us: {bank_phone}\n\n"
            "Best regards,\n{bank_name}"
        ),
        "km": (
            "Dear {name},\n\n"
            "ពាក្យសុំប្រាក់កម្ចី {ref} របស់អ្នកត្រូវបានទទួលដោយជោគជ័យ។\n\n"
            "📋 ប្រភេទ:      {loan_type}\n"
            "💰 ចំនួន:       ${amount:,.0f}\n"
            "📅 ថ្ងៃដាក់:     {date}\n\n"
            "យើងនឹងចាប់ផ្តើមពិនិត្យឯកសាររបស់អ្នកក្នុងពេលឆាប់ៗ។\n"
            "តាមដានពាក្យសុំ: http://{ip}:3000/apply/\n\n"
            "សំណួរ? ទូរស័ព្ទ: {bank_phone}\n\n"
            "{bank_name}"
        ),
        "telegram_en": (
            "📋 <b>Application Received</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 Ref:    <code>{ref}</code>\n"
            "💰 Amount: ${amount:,.0f}\n"
            "📄 Type:   {loan_type}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "We will begin reviewing your documents shortly.\n"
            "🔗 <a href='http://{ip}:3000/apply/'>Track Application</a>"
        ),
        "telegram_km": (
            "📋 <b>ពាក្យសុំត្រូវបានទទួល</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 លេខ:    <code>{ref}</code>\n"
            "💰 ចំនួន:   ${amount:,.0f}\n"
            "📄 ប្រភេទ:  {loan_type}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "យើងនឹងចាប់ផ្តើមពិនិត្យក្នុងពេលឆាប់ៗ។\n"
            "🔗 <a href='http://{ip}:3000/apply/'>តាមដានពាក្យសុំ</a>"
        ),
        "sms_en": "{bank_name}: Application {ref} received. Amount: ${amount:,.0f}. Track: http://{ip}:3000/apply/",
        "sms_km": "{bank_name}: ពាក្យសុំ {ref} ទទួលបានហើយ។ ចំនួន: ${amount:,.0f}។ តាមដាន: http://{ip}:3000/apply/",
    },
    "screening": {
        "subject_en": "Documents Under Review — {ref}",
        "subject_km": "ឯកសារកំពុងត្រូវបានពិនិត្យ — {ref}",
        "en": (
            "Dear {name},\n\n"
            "Your documents for application {ref} are now being verified "
            "by our screening system.\n\n"
            "⏱ This usually takes 1-2 business days.\n"
            "Track your application: http://{ip}:3000/apply/\n\n"
            "{bank_name}"
        ),
        "km": (
            "Dear {name},\n\n"
            "ឯកសាររបស់អ្នកសម្រាប់ពាក្យសុំ {ref} "
            "កំពុងត្រូវបានផ្ទៀងផ្ទាត់ដោយប្រព័ន្ធរបស់យើង។\n\n"
            "⏱ ជាធម្មតាចំណាយពេល ១-២ ថ្ងៃធ្វើការ។\n"
            "តាមដានពាក្យសុំ: http://{ip}:3000/apply/\n\n"
            "{bank_name}"
        ),
        "telegram_en": (
            "🔍 <b>Documents Under Review</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 Ref: <code>{ref}</code>\n"
            "⏱ Usually takes 1-2 business days.\n"
            "🔗 <a href='http://{ip}:3000/apply/'>Track Application</a>"
        ),
        "telegram_km": (
            "🔍 <b>ឯកសារកំពុងត្រូវបានពិនិត្យ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 លេខ: <code>{ref}</code>\n"
            "⏱ ជាធម្មតា ១-២ ថ្ងៃធ្វើការ។\n"
            "🔗 <a href='http://{ip}:3000/apply/'>តាមដានពាក្យសុំ</a>"
        ),
        "sms_en": "{bank_name}: Your documents for {ref} are being verified. Usually 1-2 business days.",
        "sms_km": "{bank_name}: ឯកសាររបស់អ្នក {ref} កំពុងត្រូវបានពិនិត្យ។ ជាធម្មតា ១-២ ថ្ងៃ។",
    },
    "review": {
        "subject_en": "Application Under Final Review — {ref}",
        "subject_km": "ពាក្យសុំកំពុងត្រូវបានពិនិត្យចុងក្រោយ — {ref}",
        "en": (
            "Dear {name},\n\n"
            "Your loan application {ref} is now under final review "
            "by our loan officers.\n\n"
            "One of our officers may contact you if additional "
            "information is needed.\n"
            "Track your application: http://{ip}:3000/apply/\n\n"
            "Questions? Call us: {bank_phone}\n\n"
            "{bank_name}"
        ),
        "km": (
            "Dear {name},\n\n"
            "ពាក្យសុំប្រាក់កម្ចី {ref} របស់អ្នក "
            "កំពុងស្ថិតនៅក្រោមការពិនិត្យចុងក្រោយ។\n\n"
            "មន្ត្រីរបស់យើងអាចទាក់ទងអ្នក "
            "ប្រសិនបើត្រូវការព័ត៌មានបន្ថែម។\n"
            "តាមដានពាក្យសុំ: http://{ip}:3000/apply/\n\n"
            "សំណួរ? ទូរស័ព្ទ: {bank_phone}\n\n"
            "{bank_name}"
        ),
        "telegram_en": (
            "👔 <b>Under Final Review</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 Ref: <code>{ref}</code>\n"
            "Our loan officers are reviewing your application.\n"
            "We may contact you if more info is needed.\n"
            "🔗 <a href='http://{ip}:3000/apply/'>Track Application</a>"
        ),
        "telegram_km": (
            "👔 <b>កំពុងពិនិត្យចុងក្រោយ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 លេខ: <code>{ref}</code>\n"
            "មន្ត្រីរបស់យើងកំពុងពិនិត្យពាក្យសុំរបស់អ្នក។\n"
            "🔗 <a href='http://{ip}:3000/apply/'>តាមដានពាក្យសុំ</a>"
        ),
        "sms_en": "{bank_name}: Application {ref} is under final review by our loan officers. We will contact you if needed.",
        "sms_km": "{bank_name}: ពាក្យសុំ {ref} កំពុងពិនិត្យចុងក្រោយ។ យើងនឹងទាក់ទងអ្នកប្រសិនបើត្រូវការ។",
    },
    "approved": {
        "subject_en": "🎉 Loan Approved — {ref}",
        "subject_km": "🎉 ប្រាក់កម្ចីបានអនុម័ត — {ref}",
        "en": (
            "Dear {name},\n\n"
            "Congratulations! Your loan application {ref} has been APPROVED.\n\n"
            "📋 Loan Type:     {loan_type}\n"
            "💰 Amount:        ${amount:,.0f}\n"
            "{rate_line}"
            "📝 Notes:         {notes}\n\n"
            "Our team will contact you within 2 business days "
            "to arrange the next steps.\n\n"
            "Questions? Call us: {bank_phone}\n"
            "Or email: {bank_email}\n\n"
            "Thank you for choosing {bank_name}!"
        ),
        "km": (
            "Dear {name},\n\n"
            "សូមអបអរសាទរ! ពាក្យសុំប្រាក់កម្ចី {ref} "
            "របស់អ្នកត្រូវបានអនុម័ត។\n\n"
            "📋 ប្រភេទ:        {loan_type}\n"
            "💰 ចំនួន:         ${amount:,.0f}\n"
            "{rate_line}"
            "📝 កំណត់ចំណាំ:   {notes}\n\n"
            "ក្រុមការងាររបស់យើងនឹងទាក់ទងអ្នកក្នុងរយៈពេល "
            "២ ថ្ងៃធ្វើការ ដើម្បីរៀបចំជំហានបន្ទាប់។\n\n"
            "សំណួរ? ទូរស័ព្ទ: {bank_phone}\n\n"
            "អរគុណសម្រាប់ការជ្រើសរើស {bank_name}!"
        ),
        "telegram_en": (
            "✅ <b>Loan APPROVED!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 Ref:    <code>{ref}</code>\n"
            "💰 Amount: ${amount:,.0f}\n"
            "{rate_line}"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎉 Congratulations! Our team will contact you within 2 business days.\n"
            "📞 {bank_phone}"
        ),
        "telegram_km": (
            "✅ <b>ប្រាក់កម្ចីបានអនុម័ត!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 លេខ:    <code>{ref}</code>\n"
            "💰 ចំនួន:   ${amount:,.0f}\n"
            "{rate_line}"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎉 សូមអបអរ! ក្រុមការងារនឹងទាក់ទងអ្នក ក្នុង ២ ថ្ងៃធ្វើការ។\n"
            "📞 {bank_phone}"
        ),
        "sms_en": "{bank_name}: APPROVED! Your loan {ref} for ${amount:,.0f} is approved. We contact you in 2 business days. {bank_phone}",
        "sms_km": "{bank_name}: អនុម័ត! ប្រាក់កម្ចី {ref} ចំនួន ${amount:,.0f} បានអនុម័ត។ យើងនឹងទូរស័ព្ទក្នុង ២ ថ្ងៃ។ {bank_phone}",
    },
    "rejected": {
        "subject_en": "Loan Application Update — {ref}",
        "subject_km": "ការអាប់ដេតពាក្យសុំ — {ref}",
        "en": (
            "Dear {name},\n\n"
            "Thank you for your application {ref}.\n\n"
            "After careful review, we regret to inform you that we are "
            "unable to approve your loan application at this time.\n\n"
            "{notes_line}"
            "You are welcome to reapply after 90 days.\n\n"
            "If you have any questions, please contact us:\n"
            "📞 {bank_phone}\n"
            "📧 {bank_email}\n\n"
            "{bank_name}"
        ),
        "km": (
            "Dear {name},\n\n"
            "អរគុណចំពោះពាក្យសុំ {ref} របស់អ្នក។\n\n"
            "បន្ទាប់ពីការពិនិត្យដោយយកចិត្តទុកដាក់ "
            "យើងសូមជូនដំណឹងថាមិនអាចអនុម័ត "
            "ពាក្យសុំប្រាក់កម្ចីរបស់អ្នកនាពេលនេះ។\n\n"
            "{notes_line}"
            "អ្នកអាចដាក់ពាក្យម្តងទៀតបន្ទាប់ពី ៩០ ថ្ងៃ។\n\n"
            "សំណួរ? ទូរស័ព្ទ: {bank_phone}\n\n"
            "{bank_name}"
        ),
        "telegram_en": (
            "❌ <b>Application Not Approved</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 Ref: <code>{ref}</code>\n"
            "{notes_line}"
            "You may reapply after 90 days.\n"
            "📞 {bank_phone}"
        ),
        "telegram_km": (
            "❌ <b>ពាក្យសុំមិនត្រូវបានអនុម័ត</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔖 លេខ: <code>{ref}</code>\n"
            "{notes_line}"
            "អ្នកអាចដាក់ពាក្យម្តងទៀតបន្ទាប់ពី ៩០ ថ្ងៃ។\n"
            "📞 {bank_phone}"
        ),
        "sms_en": "{bank_name}: Application {ref} was not approved at this time. Reapply after 90 days. Questions: {bank_phone}",
        "sms_km": "{bank_name}: ពាក្យសុំ {ref} មិនបានអនុម័ត។ ដាក់ពាក្យម្តងទៀតក្រោយ ៩០ ថ្ងៃ។ {bank_phone}",
    },
}


# ── Template builder ───────────────────────────────────────────────────────────

def _build(status: str, key: str, lang: str, **kwargs) -> str:
    lang = lang if lang in ("en", "km") else "en"
    tmpl = TEMPLATES.get(status, {}).get(f"{key}_{lang}",
           TEMPLATES.get(status, {}).get(f"{key}_en", ""))

    # Build optional lines
    kwargs.setdefault("bank_name",  BANK_NAME)
    kwargs.setdefault("bank_phone", BANK_PHONE)
    kwargs.setdefault("bank_email", BANK_EMAIL)
    kwargs.setdefault("ip",         SERVER_IP)
    kwargs.setdefault("date",       datetime.utcnow().strftime("%Y-%m-%d"))
    kwargs.setdefault("notes",      "")
    kwargs.setdefault("loan_type",  "")

    # Rate line
    rate = kwargs.pop("interest_rate", None)
    if rate:
        kwargs["rate_line"] = f"📈 Rate:       {rate}% p.a.\n"
    else:
        kwargs["rate_line"] = ""

    # Notes line
    notes = kwargs.get("notes", "")
    if notes:
        kwargs["notes_line"] = f"📝 Reason: {notes}\n\n"
    else:
        kwargs["notes_line"] = ""

    try:
        return tmpl.format(**kwargs)
    except KeyError as e:
        log.warning(f"Template key missing: {e}")
        return tmpl


# ═══════════════════════════════════════════════════════════════════════════════
#  CHANNEL SENDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _send_telegram(chat_id: str, message: str) -> bool:
    """Send via curl — bypasses Python SSL issues on this server."""
    if not TG_ENABLED or not BOT_TOKEN or not chat_id:
        return False
    try:
        payload = json.dumps({
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        })
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as f:
            f.write(payload)
            tmp = f.name
        r = subprocess.run([
            "curl", "-sk", "-X", "POST",
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            "-H", "Content-Type: application/json",
            "-d", f"@{tmp}",
        ], capture_output=True, text=True, timeout=15)
        os.unlink(tmp)
        result = json.loads(r.stdout)
        return result.get("ok", False)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False


def _send_sms(phone: str, message: str) -> bool:
    """Send SMS via Twilio."""
    if not SMS_ENABLED or not phone:
        return False
    try:
        from sms_notifications import _send_sms as twilio_send
        result = twilio_send(phone, message)
        return result.get("success", False)
    except Exception as e:
        log.warning(f"SMS send failed: {e}")
        return False


def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Send email via SMTP."""
    if not EMAIL_ENABLED or not to_email or not SMTP_USER:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{BANK_NAME} <{SMTP_USER}>"
        msg["To"]      = to_email

        # Plain text
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # HTML version
        html_body = body.replace("\n", "<br>")
        html = f"""
        <html><body style="font-family:Arial,sans-serif;max-width:600px;
               margin:0 auto;padding:20px;color:#333">
          <div style="background:#1a2744;padding:16px 20px;border-radius:8px 8px 0 0">
            <h2 style="color:#fff;margin:0;font-size:18px">{BANK_NAME}</h2>
          </div>
          <div style="background:#f8fafc;padding:24px;border:1px solid #e2e8f0;
               border-top:none;border-radius:0 0 8px 8px">
            <p style="white-space:pre-line;line-height:1.7">{html_body}</p>
          </div>
          <p style="font-size:11px;color:#94a3b8;margin-top:12px;text-align:center">
            {BANK_NAME} · {BANK_PHONE}
          </p>
        </body></html>"""
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, [to_email], msg.as_string())
        return True
    except Exception as e:
        log.warning(f"Email send failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN NOTIFICATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def notify_applicant(
    loan_ref:      str,
    status:        str,
    applicant_name:str = "",
    phone:         str = "",
    email:         str = "",
    telegram_id:   str = "",
    lang:          str = "en",
    loan_type:     str = "",
    loan_amount:   float = 0,
    notes:         str = "",
    interest_rate  = None,
) -> dict:
    """
    Send status update to applicant via all available channels.

    status: submitted | screening | review | approved | rejected

    Returns dict with results from each channel.
    """
    if status not in TEMPLATES:
        return {"error": f"unknown status: {status}"}

    kwargs = dict(
        ref           = loan_ref,
        name          = applicant_name or "Valued Customer",
        loan_type     = loan_type,
        amount        = float(loan_amount or 0),
        notes         = notes or "",
        interest_rate = interest_rate,
    )

    results = {}

    # ── Telegram ──────────────────────────────────────────────────────────────
    if telegram_id:
        msg = _build(status, "telegram", lang, **kwargs)
        results["telegram"] = _send_telegram(telegram_id, msg)
        log.info(f"Telegram → {telegram_id}: {results['telegram']}")
    else:
        results["telegram"] = False

    # ── SMS ───────────────────────────────────────────────────────────────────
    if phone:
        msg = _build(status, "sms", lang, **kwargs)
        results["sms"] = _send_sms(phone, msg)
        log.info(f"SMS → {phone}: {results['sms']}")
    else:
        results["sms"] = False

    # ── Email ─────────────────────────────────────────────────────────────────
    if email:
        subject = _build(status, "subject", lang, **kwargs)
        body    = _build(status, lang,       lang, **kwargs)
        results["email"] = _send_email(email, subject, body)
        log.info(f"Email → {email}: {results['email']}")
    else:
        results["email"] = False

    # ── Queue for staff Telegram bridge ───────────────────────────────────────
    try:
        _queue_for_bridge(loan_ref, status, applicant_name,
                          loan_amount, loan_type, results)
    except Exception as e:
        log.warning(f"Bridge queue failed: {e}")

    return results


def _queue_for_bridge(loan_ref, status, name, amount, loan_type, channel_results):
    """Also notify staff via the Telegram bridge queue."""
    status_emoji = {
        "submitted": "📋", "screening": "🔍", "review": "👔",
        "approved":  "✅", "rejected":  "❌",
    }
    status_label = {
        "submitted": "Application Received",
        "screening": "Screening Started",
        "review":    "Under Review",
        "approved":  "APPROVED",
        "rejected":  "Rejected",
    }
    emoji = status_emoji.get(status, "📋")
    label = status_label.get(status, status.title())

    ch_icons = []
    if channel_results.get("telegram"): ch_icons.append("💬TG")
    if channel_results.get("sms"):      ch_icons.append("📱SMS")
    if channel_results.get("email"):    ch_icons.append("📧Email")
    sent_via = " ".join(ch_icons) if ch_icons else "none"

    msg = (
        f"{emoji} <b>Applicant Notified: {label}</b>\n"
        f"🔖 Ref:  <code>{loan_ref}</code>\n"
        f"👤 Name: {name or '—'}\n"
        f"💰 Amt:  ${float(amount or 0):,.0f}\n"
        f"📤 Sent: {sent_via}"
    )
    db = _db()
    try:
        db.execute(text(
            "INSERT INTO telegram_queue (message) VALUES (:m)"
        ), {"m": msg})
        db.commit()
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM BOT — APPLICANT ONBOARDING
# ═══════════════════════════════════════════════════════════════════════════════

def init_applicant_telegram_table():
    """Create table to store applicant Telegram chat IDs."""
    db = _db()
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS applicant_telegram (
                id         SERIAL PRIMARY KEY,
                chat_id    VARCHAR(50) UNIQUE,
                loan_ref   VARCHAR(50),
                username   VARCHAR(100),
                linked_at  TIMESTAMP DEFAULT NOW()
            )
        """))
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"Init table: {e}")
    finally:
        db.close()


def get_applicant_telegram_id(loan_ref: str) -> str:
    """Get Telegram chat_id for a loan reference."""
    db = _db()
    try:
        row = db.execute(text(
            "SELECT chat_id FROM applicant_telegram WHERE loan_ref=:r"
        ), {"r": loan_ref}).fetchone()
        return row[0] if row else ""
    except: return ""
    finally: db.close()


def link_telegram_to_loan(chat_id: str, loan_ref: str, username: str = "") -> bool:
    """Link a Telegram chat_id to a loan reference."""
    db = _db()
    try:
        db.execute(text("""
            INSERT INTO applicant_telegram (chat_id, loan_ref, username)
            VALUES (:c, :r, :u)
            ON CONFLICT (chat_id) DO UPDATE SET loan_ref=:r, linked_at=NOW()
        """), {"c": chat_id, "r": loan_ref, "u": username})
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        log.error(f"Link telegram: {e}")
        return False
    finally:
        db.close()


def process_bot_updates():
    """
    Process incoming Telegram bot messages from applicants.
    Call this in a loop to handle /start and loan ref linking.

    Applicant flow:
      1. Applicant starts: /start
      2. Bot asks for loan reference
      3. Applicant sends: AP2604173144
      4. Bot links their chat_id to that loan ref
      5. All future updates sent directly to them
    """
    try:
        r = subprocess.run([
            "curl", "-sk",
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            "--data", "timeout=5",
        ], capture_output=True, text=True, timeout=20)
        data    = json.loads(r.stdout)
        updates = data.get("result", [])

        last_update_id = None
        for update in updates:
            last_update_id = update.get("update_id")
            msg  = update.get("message", {})
            text = msg.get("text", "").strip()
            chat = msg.get("chat", {})
            chat_id  = str(chat.get("id", ""))
            username = chat.get("username", "")
            fname    = chat.get("first_name", "")

            if not text or not chat_id:
                continue

            if text.startswith("/start"):
                _send_telegram(chat_id,
                    f"👋 <b>Welcome to {BANK_NAME} Loan Tracker!</b>\n\n"
                    f"Send your loan reference number to receive updates.\n"
                    f"Example: <code>AP2604173144</code>"
                )

            elif re.match(r"^AP\d{8,12}$", text.upper()):
                loan_ref = text.upper()
                # Verify loan exists
                db = _db()
                try:
                    row = db.execute(text(
                        "SELECT applicant_name, status FROM applicant_loans WHERE loan_ref=:r"
                    ), {"r": loan_ref}).fetchone()
                finally:
                    db.close()

                if row:
                    link_telegram_to_loan(chat_id, loan_ref, username)
                    status_map = {
                        "draft":     "Draft",
                        "submitted": "Submitted ✓",
                        "screening": "Under Screening 🔍",
                        "review":    "Under Review 👔",
                        "approved":  "Approved ✅",
                        "rejected":  "Not Approved ❌",
                    }
                    current = status_map.get(row[1], row[1].title())
                    _send_telegram(chat_id,
                        f"✅ <b>Linked!</b>\n\n"
                        f"🔖 Ref:    <code>{loan_ref}</code>\n"
                        f"👤 Name:   {row[0] or '—'}\n"
                        f"📊 Status: {current}\n\n"
                        f"You will receive automatic updates when your application status changes."
                    )
                else:
                    _send_telegram(chat_id,
                        f"❌ Loan reference <code>{loan_ref}</code> not found.\n"
                        f"Please check your reference number and try again."
                    )
            else:
                _send_telegram(chat_id,
                    f"Please send your loan reference number.\n"
                    f"Format: <code>AP</code> followed by numbers.\n"
                    f"Example: <code>AP2604173144</code>"
                )

        # Mark updates as processed
        if last_update_id:
            subprocess.run([
                "curl", "-sk",
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                "--data", f"offset={last_update_id + 1}",
            ], capture_output=True, timeout=10)

        return len(updates)

    except Exception as e:
        log.error(f"Bot update processing failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: notify from loan_ref only (auto-lookup applicant details)
# ═══════════════════════════════════════════════════════════════════════════════

def notify_by_ref(loan_ref: str, status: str,
                  notes: str = "", interest_rate=None) -> dict:
    """
    Convenience function — looks up all applicant details from loan_ref
    and sends notifications via all available channels automatically.

    Call from anywhere in the system:
      from applicant_notifications import notify_by_ref
      notify_by_ref("AP2604173144", "approved", notes="Approved at 12% p.a.")
    """
    db = _db()
    try:
        # Get loan details
        loan = db.execute(text("""
            SELECT al.loan_ref, al.loan_type, al.loan_amount,
                   a.name, a.phone, a.email
            FROM applicant_loans al
            LEFT JOIN applicants a ON a.id = al.applicant_id
            WHERE al.loan_ref = :r
        """), {"r": loan_ref}).fetchone()

        if not loan:
            # Try loan_applications table
            loan = db.execute(text("""
                SELECT loan_ref, loan_type, loan_amount,
                       applicant_name, applicant_phone, applicant_email
                FROM loan_applications WHERE loan_ref = :r
            """), {"r": loan_ref}).fetchone()

        if not loan:
            return {"error": f"loan_ref {loan_ref} not found"}

        # Get Telegram chat_id if linked
        tg_id = get_applicant_telegram_id(loan_ref)

        return notify_applicant(
            loan_ref       = loan_ref,
            status         = status,
            applicant_name = loan[3] or "",
            phone          = loan[4] or "",
            email          = loan[5] or "",
            telegram_id    = tg_id,
            loan_type      = loan[1] or "",
            loan_amount    = float(loan[2] or 0),
            notes          = notes,
            interest_rate  = interest_rate,
            lang           = "en",
        )
    except Exception as e:
        log.error(f"notify_by_ref failed: {e}")
        return {"error": str(e)}
    finally:
        db.close()


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    init_applicant_telegram_table()

    if len(sys.argv) >= 3:
        ref    = sys.argv[1]
        status = sys.argv[2]
        print(f"Testing notification: {ref} → {status}")
        result = notify_by_ref(ref, status)
        print("Result:", json.dumps(result, indent=2))
    else:
        print("Usage: python applicant_notifications.py <loan_ref> <status>")
        print("Example: python applicant_notifications.py AP2604173144 approved")
        print("\nStatuses: submitted | screening | review | approved | rejected")
        print("\nChannel status:")
        print(f"  Telegram : {'ENABLED' if TG_ENABLED else 'DISABLED'}")
        print(f"  SMS      : {'ENABLED' if SMS_ENABLED else 'DISABLED'}")
        print(f"  Email    : {'ENABLED' if EMAIL_ENABLED else 'DISABLED'}")
