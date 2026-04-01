"""
celery_app.py — Celery + Redis broker configuration
"""
import os
import sys
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# ── Ensure /app is in Python path for all forked worker subprocesses ──────────
# Without this, imports like face_match, hologram, template_match fail
# in Celery's forked processes even though the files exist in /app
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "fraud_detect",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                        # re-queue on worker crash
    worker_prefetch_multiplier=1,               # fair dispatch for heavy CV tasks
    result_expires=3600,                        # results kept 1 hour
    worker_send_task_events=True,               # enable task events for Flower
    task_send_sent_event=True,                  # track when tasks are sent
    broker_connection_retry_on_startup=True,    # suppress deprecation warning
    task_routes={
        "tasks.screen_pdf_task":     {"queue": "pdf"},
        "tasks.screen_image_task":   {"queue": "image"},
        "tasks.screen_id_card_task": {"queue": "idcard"},
    },
)
