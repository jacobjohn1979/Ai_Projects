# Banking Document Fraud Detection API — v3.0

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI  (app.py)                     │
│  POST /screen-pdf   POST /screen-image   POST /screen-id-card│
│  GET  /result/{id}  GET  /history/{id_number}                │
└────────────────────────┬────────────────────────────────────┘
                         │  submit task (202 Accepted)
                         ▼
              ┌──────────────────────┐
              │   Redis  (broker)    │  ←── task queue
              └──────────┬───────────┘
                         │  consume
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
   [pdf worker]   [image worker]    [idcard worker]
   tasks.py        tasks.py          tasks.py
          │              │                  │
          ▼              ▼                  ▼
   inspect_pdf_   extract_image_    extract_id_card_
   forensics()    metadata()        fields()
   font_check()   ELA()             MRZ checksum
   redaction()    noise_analysis()  region sharpness
   JS detect()    copy_move()       color temperature
   field_checks() basic_forensics() noise fingerprint
                                    hologram.py ───────┐
                                    template_match.py  │
                                    face_match.py ─────┘
          │              │                  │
          └──────────────┴──────────────────┘
                         │  save result
                         ▼
              ┌──────────────────────┐
              │   PostgreSQL         │
              │  screening_logs      │
              │  velocity_events     │
              └──────────────────────┘
```

---

## New Features in v3

### 1. Face Liveness + Match (`face_match.py`)
- **Liveness score (0–100)** via LBP texture, specular highlights, FFT frequency analysis, and blur variance
- Flags: `likely_spoofed_face` (score <30), `possible_spoofed_face` (score <50)
- **ArcFace face comparison** (DeepFace, local — no cloud) between selfie and ID card photo
- Flags: `face_match_failed`, `low_face_similarity`
- Upload selfie via `selfie=` field on `/screen-id-card`

### 2. Hologram & Security Feature Detection (`hologram.py`)
| Check | Method |
|---|---|
| Iridescence (hue variance) | HSV analysis of high-saturation pixels |
| Security background pattern | FFT peak-to-mean ratio |
| Micro-text density | High-pass kernel pixel count |
| Holographic patch | Localised brightness anomaly (16 blocks) |
| Combined absence | `multiple_security_features_absent` if ≥2 checks fail |

### 3. Document Template Matching (`template_match.py`)
- Classifies document by MRZ country code + OCR keywords
- Built-in templates: Generic Passport (TD3), Generic ID Card (TD1), Visa (MRVA), Cambodia (KHM), Thailand (THA)
- Validates: MRZ line count, MRZ line length, aspect ratio (±15%), zone presence, background colour, keyword coverage
- Easily extensible — add a new `IDTemplate` entry to `TEMPLATES` dict

### 4. Velocity Checks (`database.py`)
| Check | Trigger |
|---|---|
| `duplicate_file_resubmission` | Same SHA-256 submitted ≥ `MAX_SAME_FILE_SUBMISSIONS` times in window |
| `id_number_velocity_breach` | Same ID number submitted ≥ `MAX_SUBMISSIONS_PER_ID` times in window |
| `id_previously_flagged_high_risk` | ID number has any prior HIGH-risk result ever |

Configure via `.env`:
```
VELOCITY_WINDOW_HOURS=24
MAX_SUBMISSIONS_PER_ID=3
MAX_SAME_FILE_SUBMISSIONS=2
```

### 5. Async Celery + Redis (`celery_app.py`, `tasks.py`)
- All three endpoints return `{"task_id": "...", "status": "queued"}` immediately (202)
- Poll `GET /result/{task_id}` for status: `pending` → `processing` → `complete` | `failed`
- Three dedicated queues: `pdf`, `image`, `idcard`
- Tasks retry up to 2× on failure with backoff
- Results stored in Redis for 1 hour

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit environment
cp .env.example .env

# 3. Create PostgreSQL database
createdb fraud_detect

# 4. Start Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine

# 5. Start Celery workers (3 queues)
celery -A celery_app worker -Q pdf     --concurrency 2 --loglevel info &
celery -A celery_app worker -Q image   --concurrency 2 --loglevel info &
celery -A celery_app worker -Q idcard  --concurrency 1 --loglevel info &

# 6. Start FastAPI
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

---

## API Usage

### Submit ID card + selfie
```bash
curl -X POST http://localhost:8001/screen-id-card \
  -F "file=@id_card.jpg" \
  -F "selfie=@selfie.jpg" \
  -F "applicant_id=APP-12345"
# → {"task_id": "abc123", "status": "queued"}
```

### Poll for result
```bash
curl http://localhost:8001/result/abc123
# → {"task_id": "abc123", "status": "complete", "result": {...}}
```

### Check submission history
```bash
curl http://localhost:8001/history/123456789
```

---

## Flag Weight Reference (ID Card — score ≥50 = HIGH)

| Flag | Score |
|---|---|
| `face_match_failed` | +40 |
| `likely_spoofed_face` | +40 |
| `multiple_security_features_absent` | +35 |
| `copy_move_detected` | +35 |
| `id_number_mrz_mismatch` | +35 |
| `id_previously_flagged_high_risk` | +35 |
| `mrz_dob_checksum_failed` | +30 |
| `inconsistent_noise_pattern` | +30 |
| `id_number_velocity_breach` | +25 |
| `suspicious_editing_software` | +25 |
| `color_temperature_mismatch` | +25 |
| `noise_fingerprint_mismatch` | +25 |
| `low_face_similarity` | +30 |
| `no_iridescence_detected` | +20 |
| `missing_security_background_pattern` | +20 |

---

## Adding New ID Templates

Edit `template_match.py` → `TEMPLATES` dict:

```python
"MYS_ID": IDTemplate(
    country_code = "MYS",
    country_name = "Malaysia MyKad",
    doc_types    = ["IDENTITY CARD", "MYKAD"],
    mrz_type     = "TD1",
    mrz_lines    = 3,
    mrz_line_len = 30,
    aspect_ratio = 1.585,
    bg_hsv       = (15, 50, 220),
    keywords     = ["MALAYSIA", "WARGANEGARA", "IDENTITY"],
    zones        = [
        ("photo", 0.00, 0.00, 0.30, 0.60),
        ("mrz",   0.00, 0.65, 1.00, 0.35),
    ],
),
```

Then add `"MYS": "MYS_ID"` to `MRZ_COUNTRY_MAP`.
