"""telegram_bridge.py — Runs on HOST server.
- Sends queued Telegram alerts from telegram_queue
- Handles applicant Telegram replies
- Links Telegram chat_id to loan_ref
- Collects phone number if missing
"""

import os, time, re, subprocess, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('/home/docuser/apps/Ai_Projects/backend/.env')

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLED   = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"
BANK_NAME = os.getenv("BANK_NAME", "Bank")

# Get Postgres container IP
result = subprocess.run(
    [
        "docker", "inspect", "backend-postgres-1",
        "--format", "{{.NetworkSettings.Networks.backend_internal.IPAddress}}"
    ],
    capture_output=True,
    text=True
)

PG_HOST = result.stdout.strip() or "172.20.0.2"
print(f"Postgres IP: {PG_HOST}")

# Parse DB credentials
DB_URL = os.getenv("DATABASE_URL", "")
m = re.match(r"postgresql://([^:]+):([^@]+)@[^:]+:\d+/(.+)", DB_URL)

if m:
    PG_USER, PG_PASS, PG_DB = m.group(1), m.group(2), m.group(3)
    PG_PASS = requests.utils.unquote(PG_PASS)
else:
    print("ERROR: Could not parse DATABASE_URL")
    exit(1)

print(f"DB: {PG_USER}@{PG_HOST}/{PG_DB}")


def get_db():
    import psycopg2
    return psycopg2.connect(
        host=PG_HOST,
        port=5432,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS
    )


def send_via_curl(msg, chat_id=None):
    if not ENABLED or not BOT_TOKEN:
        return False

    target_chat = chat_id or CHAT_ID

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": target_chat,
                "text": msg,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=10,
            verify=False
        )

        result = r.json()

        if result.get("ok"):
            print(f"  ✓ Sent to {target_chat}: {msg[:50]}...")
            return True
        else:
            print(f"  ✗ Telegram error: {result.get('description')}")
            return False

    except Exception as e:
        print(f"  ✗ Send failed: {e}")
        return False


def send(msg):
    return send_via_curl(msg, CHAT_ID)


def process_queue(conn):
    with conn.cursor() as c:
        c.execute("""
            SELECT id, message FROM telegram_queue
            WHERE sent=FALSE
            ORDER BY created_at
            LIMIT 10
        """)
        rows = c.fetchall()

        for row_id, message in rows:
            if send(message):
                c.execute("""
                    UPDATE telegram_queue
                    SET sent=TRUE, sent_at=NOW()
                    WHERE id=%s
                """, (row_id,))

        conn.commit()
        return len(rows)


def get_updates(offset=None):
    if not ENABLED or not BOT_TOKEN:
        return []

    try:
        params = {
            "timeout": 1,
            "allowed_updates": ["message"]
        }

        if offset:
            params["offset"] = offset

        r = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params=params,
            timeout=5,
            verify=False
        )

        data = r.json()
        if data.get("ok"):
            return data.get("result", [])

    except Exception as e:
        print(f"Get updates error: {e}")

    return []


def handle_message(conn, message):
    chat_id = str(message.get("chat", {}).get("id", ""))
    text_in = (message.get("text") or "").strip()
    fname = message.get("from", {}).get("first_name", "")

    if not chat_id or not text_in:
        return

    print(f"Incoming from {chat_id}: {text_in}")

    if text_in.lower() == "/start":
        reply = (
            f"👋 Welcome to <b>{BANK_NAME}</b> loan update service.\n\n"
            f"Please send your loan reference number.\n"
            f"It starts with <b>AP</b> followed by numbers.\n"
            f"Example: <code>AP2604173144</code>"
        )
        send_via_curl(reply, chat_id)
        return

    # Loan reference linking
    if re.match(r"^AP\d{6,20}$", text_in.upper()):
        loan_ref = text_in.upper()

        try:
            with conn.cursor() as c:
                c.execute("""
                    SELECT al.loan_ref
                    FROM applicant_loans al
                    WHERE al.loan_ref=%s
                    LIMIT 1
                """, (loan_ref,))
                row = c.fetchone()

            if row:
                # Ask phone if missing
                try:
                    with conn.cursor() as c:
                        c.execute("""
                            SELECT a.phone
                            FROM applicants a
                            JOIN applicant_loans al ON al.applicant_id = a.id
                            WHERE al.loan_ref=%s
                            LIMIT 1
                        """, (loan_ref,))
                        phone_row = c.fetchone()

                        if phone_row and not phone_row[0]:
                            send_via_curl(
                                "Please share your phone number for SMS updates.\n"
                                "Send in format: <code>+855xxxxxxxxx</code>\n"
                                "Or send <b>skip</b> to continue without SMS.",
                                chat_id
                            )
                except Exception as pe:
                    print(f"Phone check error: {pe}")

                # Link Telegram chat to loan ref
                try:
                    with conn.cursor() as c:
                        c.execute("""
                            INSERT INTO applicant_telegram
                            (chat_id, loan_ref, username)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (chat_id) DO UPDATE
                            SET loan_ref=%s, linked_at=NOW()
                        """, (chat_id, loan_ref, fname, loan_ref))

                    conn.commit()

                    send_via_curl(
                        f"✅ Your Telegram has been linked to loan reference "
                        f"<b>{loan_ref}</b>.\n\n"
                        f"You will receive updates here.",
                        chat_id
                    )

                except Exception as le:
                    print(f"Link error: {le}")
                    try:
                        conn.rollback()
                    except:
                        pass

            else:
                send_via_curl(
                    f"❌ Loan reference <b>{loan_ref}</b> was not found.\n"
                    f"Please check and send again.",
                    chat_id
                )

        except Exception as e:
            print(f"Loan ref check error: {e}")
            try:
                conn.rollback()
            except:
                pass

            send_via_curl(
                "System could not check your loan reference now. Please try again.",
                chat_id
            )

        return

    # Phone number collection
    if re.match(r"^\+\d{8,15}$", text_in):
        phone = text_in

        try:
            with conn.cursor() as c:
                c.execute("""
                    SELECT loan_ref
                    FROM applicant_telegram
                    WHERE chat_id=%s
                    LIMIT 1
                """, (chat_id,))
                tr = c.fetchone()

                if tr:
                    loan_ref = tr[0]

                    c.execute("""
                        UPDATE applicants
                        SET phone=%s
                        WHERE id = (
                            SELECT applicant_id
                            FROM applicant_loans
                            WHERE loan_ref=%s
                            LIMIT 1
                        )
                    """, (phone, loan_ref))

                    c.execute("""
                        UPDATE loan_applications
                        SET applicant_phone=%s
                        WHERE loan_ref=%s
                    """, (phone, loan_ref))

                    conn.commit()

                    send_via_curl(
                        f"✅ Phone <code>{phone}</code> saved.\n"
                        f"You will receive SMS updates on this number.",
                        chat_id
                    )
                else:
                    send_via_curl(
                        "Please send your loan reference number first.",
                        chat_id
                    )

        except Exception as pe:
            print(f"Phone save error: {pe}")
            try:
                conn.rollback()
            except:
                pass

            send_via_curl(
                "Could not save phone. Please try again.",
                chat_id
            )

        return

    if text_in.lower() == "skip":
        send_via_curl(
            "OK — no SMS updates. You will still receive Telegram updates here.",
            chat_id
        )
        return

    reply = (
        f"Please send your loan reference number.\n"
        f"It starts with <b>AP</b> followed by numbers.\n"
        f"Example: <code>AP2604173144</code>\n\n"
        f"Or type /start to begin."
    )
    send_via_curl(reply, chat_id)


def process_incoming_messages(conn, last_update_id):
    updates = get_updates(last_update_id + 1 if last_update_id else None)

    for upd in updates:
        update_id = upd.get("update_id")
        if update_id:
            last_update_id = update_id

        msg = upd.get("message")
        if msg:
            handle_message(conn, msg)

    return last_update_id


print("Starting Telegram bridge — polling every 5 seconds")
print(f"Token: {BOT_TOKEN[:20]}...")
print(f"Chat:  {CHAT_ID}")
print(f"Enabled: {ENABLED}")

try:
    conn = get_db()
    print("DB connected OK")
except Exception as e:
    print(f"ERROR connecting to DB: {e}")
    exit(1)

send(
    f"✅ <b>Telegram Bridge Active</b>\n"
    f"🏦 {BANK_NAME} KYC\n"
    f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
)

print("Bridge running — press Ctrl+C to stop\n")

last_update_id = 0

while True:
    try:
        n = process_queue(conn)
        if n:
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Processed {n} alerts")

        last_update_id = process_incoming_messages(conn, last_update_id)

    except Exception as e:
        print(f"Error: {e} — reconnecting...")
        try:
            conn.close()
        except:
            pass

        try:
            conn = get_db()
        except Exception as re:
            print(f"Reconnect failed: {re}")

    time.sleep(5)