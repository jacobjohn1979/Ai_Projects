"""
sms_notifications.py — SMS Notification System via Twilio
Sends bilingual (Khmer + English) SMS notifications to loan applicants.

Events:
  - Application received
  - KYC screening started
  - Approved (with loan amount and next steps)
  - Rejected (with reason)
  - Review needed (officer will contact)
  - OTP verification codes

Setup:
  Add to .env:
    SMS_ENABLED=true
    TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TWILIO_AUTH_TOKEN=your_auth_token
    TWILIO_FROM_NUMBER=+14155550100
    BANK_NAME=Your Bank Name
    BANK_PHONE=+855 xx xxx xxx
"""

import os
import logging
import urllib.request
import urllib.parse
import base64
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("sms")

SMS_ENABLED     = os.getenv("SMS_ENABLED", "false").lower() == "true"
ACCOUNT_SID     = os.getenv("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN", "")
FROM_NUMBER     = os.getenv("TWILIO_FROM_NUMBER", "")
BANK_NAME       = os.getenv("BANK_NAME", "Bank")
BANK_PHONE      = os.getenv("BANK_PHONE", "")
PORTAL_URL      = os.getenv("APPLICANT_PORTAL_URL", "")
SERVER_IP       = os.getenv("SERVER_IP", "172.16.26.48")

# ── Message templates (Khmer + English) ──────────────────────────────────────

MESSAGES = {
    "received": {
        "en": (
            "{bank_name}: Your loan application {ref} has been received. "
            "We will begin processing your documents shortly. "
            "Track your status: http://{ip}:3000/apply/"
        ),
        "km": (
            "{bank_name}: ពាក្យសុំប្រាក់កម្ចី {ref} របស់អ្នកត្រូវបានទទួល។ "
            "យើងនឹងចាប់ផ្តើមដំណើរការឯកសាររបស់អ្នក។ "
            "តាមដានស្ថានភាព: http://{ip}:3000/apply/"
        ),
    },
    "screening": {
        "en": (
            "{bank_name}: Your documents for application {ref} are now "
            "being reviewed by our verification team. "
            "This usually takes 1-2 business days."
        ),
        "km": (
            "{bank_name}: ឯកសាររបស់អ្នកសម្រាប់ពាក្យសុំ {ref} "
            "កំពុងត្រូវបានពិនិត្យ។ "
            "ជាធម្មតាចំណាយពេល ១-២ ថ្ងៃធ្វើការ។"
        ),
    },
    "approved": {
        "en": (
            "{bank_name}: Congratulations! Your loan application {ref} "
            "for {currency}{amount} has been APPROVED. "
            "Our team will contact you within 2 business days to arrange next steps. "
            "{notes}"
            "Questions? Call us: {bank_phone}"
        ),
        "km": (
            "{bank_name}: សូមអបអរសាទរ! ពាក្យសុំប្រាក់កម្ចី {ref} "
            "ចំនួន {currency}{amount} របស់អ្នកត្រូវបានអនុម័ត។ "
            "ក្រុមការងាររបស់យើងនឹងទាក់ទងអ្នកក្នុងរយៈពេល ២ ថ្ងៃធ្វើការ។ "
            "{notes}"
            "សំណួរ? ទូរស័ព្ទ: {bank_phone}"
        ),
    },
    "rejected": {
        "en": (
            "{bank_name}: We regret to inform you that your loan application {ref} "
            "could not be approved at this time. "
            "{notes}"
            "You may reapply after 90 days. "
            "Questions? Call us: {bank_phone}"
        ),
        "km": (
            "{bank_name}: យើងសូមជូនដំណឹងថាពាក្យសុំប្រាក់កម្ចី {ref} "
            "របស់អ្នកមិនអាចអនុម័តបាននាពេលនេះ។ "
            "{notes}"
            "អ្នកអាចដាក់ពាក្យម្តងទៀតបន្ទាប់ពី ៩០ ថ្ងៃ។ "
            "សំណួរ? ទូរស័ព្ទ: {bank_phone}"
        ),
    },
    "review": {
        "en": (
            "{bank_name}: Your loan application {ref} requires additional review. "
            "One of our officers will contact you within 1 business day. "
            "Questions? Call us: {bank_phone}"
        ),
        "km": (
            "{bank_name}: ពាក្យសុំប្រាក់កម្ចី {ref} របស់អ្នកត្រូវការពិនិត្យបន្ថែម។ "
            "មន្ត្រីរបស់យើងនឹងទំនាក់ទំនងអ្នកក្នុងរយៈពេល ១ ថ្ងៃធ្វើការ។ "
            "សំណួរ? ទូរស័ព្ទ: {bank_phone}"
        ),
    },
    "otp": {
        "en": (
            "{bank_name}: Your verification code is {otp}. "
            "Valid for 10 minutes. Do not share this code."
        ),
        "km": (
            "{bank_name}: លេខផ្ទៀងផ្ទាត់របស់អ្នកគឺ {otp}។ "
            "មានសុពលភាព ១០ នាទី។ "
            "កុំចែករំលែកលេខនេះ។"
        ),
    },
}


# ── Core SMS sender ───────────────────────────────────────────────────────────

def _send_sms(to_number: str, body: str) -> dict:
    """
    Send SMS via Twilio REST API.
    Returns {"success": True, "sid": "..."} or {"success": False, "error": "..."}
    """
    if not SMS_ENABLED:
        log.info(f"SMS disabled — would send to {to_number}: {body[:60]}...")
        return {"success": False, "reason": "SMS_ENABLED=false"}

    if not all([ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER]):
        log.error("Twilio credentials not configured in .env")
        return {"success": False, "error": "credentials_missing"}

    # Normalise phone number
    to_clean = _normalise_phone(to_number)
    if not to_clean:
        return {"success": False, "error": f"invalid_phone: {to_number}"}

    try:
        url  = f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT_SID}/Messages.json"
        data = urllib.parse.urlencode({
            "To":   to_clean,
            "From": FROM_NUMBER,
            "Body": body,
        }).encode("utf-8")

        creds = base64.b64encode(f"{ACCOUNT_SID}:{AUTH_TOKEN}".encode()).decode()
        req   = urllib.request.Request(url, data=data, method="POST", headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        })

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            sid    = result.get("sid", "")
            status = result.get("status", "")
            log.info(f"SMS sent to {to_clean} — SID: {sid} Status: {status}")
            return {"success": True, "sid": sid, "status": status}

    except urllib.error.HTTPError as e:
        body_err = e.read().decode()
        log.error(f"Twilio HTTP error {e.code}: {body_err}")
        return {"success": False, "error": f"twilio_{e.code}", "detail": body_err}
    except Exception as e:
        log.error(f"SMS send failed: {e}")
        return {"success": False, "error": str(e)}


def _normalise_phone(phone: str) -> str:
    """Normalise phone number to E.164 format for Cambodia."""
    if not phone:
        return ""
    import re
    digits = re.sub(r"[^\d+]", "", phone)

    # Already E.164
    if digits.startswith("+"):
        return digits

    # Cambodia local format: 0xx → +855xx
    if digits.startswith("0") and len(digits) >= 9:
        return "+855" + digits[1:]

    # Bare Cambodia number without leading 0
    if len(digits) == 8 or len(digits) == 9:
        return "+855" + digits.lstrip("0")

    return digits if digits else ""


def _build_message(event: str, lang: str, **kwargs) -> str:
    """Build localised SMS message from template."""
    lang     = lang if lang in ("en", "km") else "en"
    template = MESSAGES.get(event, {}).get(lang, MESSAGES.get(event, {}).get("en", ""))

    defaults = {
        "bank_name":  BANK_NAME,
        "bank_phone": BANK_PHONE,
        "ip":         SERVER_IP,
        "currency":   "$",
        "notes":      "",
        "otp":        "",
        "ref":        "",
        "amount":     "",
    }
    defaults.update(kwargs)

    # Format amount with commas
    if defaults.get("amount"):
        try:
            defaults["amount"] = f"{float(defaults['amount']):,.0f}"
        except (ValueError, TypeError):
            pass

    return template.format(**defaults)


# ── Public notification functions ─────────────────────────────────────────────

def notify_received(phone: str, loan_ref: str, lang: str = "en") -> dict:
    """Notify applicant their application was received."""
    body = _build_message("received", lang, ref=loan_ref)
    return _send_sms(phone, body)


def notify_screening(phone: str, loan_ref: str, lang: str = "en") -> dict:
    """Notify applicant their documents are being screened."""
    body = _build_message("screening", lang, ref=loan_ref)
    return _send_sms(phone, body)


def notify_approved(
    phone:      str,
    loan_ref:   str,
    amount:     float,
    lang:       str = "en",
    notes:      str = "",
    currency:   str = "$",
) -> dict:
    """Notify applicant their loan was approved."""
    note_text = f"Note: {notes} " if notes else ""
    body = _build_message("approved", lang,
                          ref=loan_ref, amount=amount,
                          currency=currency, notes=note_text)
    return _send_sms(phone, body)


def notify_rejected(
    phone:    str,
    loan_ref: str,
    lang:     str = "en",
    notes:    str = "",
) -> dict:
    """Notify applicant their loan was not approved."""
    note_text = f"Reason: {notes} " if notes else ""
    body = _build_message("rejected", lang, ref=loan_ref, notes=note_text)
    return _send_sms(phone, body)


def notify_review(phone: str, loan_ref: str, lang: str = "en") -> dict:
    """Notify applicant their application needs manual review."""
    body = _build_message("review", lang, ref=loan_ref)
    return _send_sms(phone, body)


def send_otp(phone: str, otp: str, lang: str = "en") -> dict:
    """Send OTP verification code."""
    body = _build_message("otp", lang, otp=otp)
    return _send_sms(phone, body)


# ── Batch: notify by loan status change ──────────────────────────────────────

def notify_status_change(
    phone:    str,
    loan_ref: str,
    status:   str,
    lang:     str = "en",
    amount:   float = 0,
    notes:    str = "",
) -> dict:
    """
    Single function to call on any status change.
    status: submitted | screening | approved | rejected | review
    """
    if not phone:
        return {"success": False, "reason": "no_phone"}

    dispatch = {
        "submitted": lambda: notify_received(phone, loan_ref, lang),
        "screening": lambda: notify_screening(phone, loan_ref, lang),
        "approved":  lambda: notify_approved(phone, loan_ref, amount, lang, notes),
        "rejected":  lambda: notify_rejected(phone, loan_ref, lang, notes),
        "review":    lambda: notify_review(phone, loan_ref, lang),
    }

    fn = dispatch.get(status)
    if not fn:
        return {"success": False, "reason": f"no_template_for_{status}"}

    try:
        result = fn()
        log.info(f"Notification {status} → {phone} for {loan_ref}: {result}")
        return result
    except Exception as e:
        log.error(f"notify_status_change failed: {e}")
        return {"success": False, "error": str(e)}


# ── Test utility ──────────────────────────────────────────────────────────────

def test_sms(to_number: str, lang: str = "en") -> dict:
    """
    Send a test SMS to verify configuration.
    Run: python sms_notifications.py +855xxxxxxxxx
    """
    body = (
        f"{BANK_NAME}: SMS notification test successful. "
        f"Your system is configured correctly. "
        f"Time: {datetime.utcnow().strftime('%H:%M UTC')}"
        if lang == "en" else
        f"{BANK_NAME}: ការបញ្ជូន SMS សាកល្បងបានជោគជ័យ។ "
        f"ប្រព័ន្ធរបស់អ្នកត្រូវបានកំណត់រចនាសម្ព័ន្ធត្រឹមត្រូវ។"
    )
    return _send_sms(to_number, body)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sms_notifications.py <phone_number> [lang]")
        print("Example: python sms_notifications.py +855123456789 km")
        sys.exit(1)

    phone = sys.argv[1]
    lang  = sys.argv[2] if len(sys.argv) > 2 else "en"

    print(f"Testing SMS to {phone} in {lang}...")
    result = test_sms(phone, lang)
    print("Result:", json.dumps(result, indent=2))
