# Docker Deployment Guide

## File structure after adding Docker files

```
fraud_detection/
├── app.py                  ← FastAPI app
├── tasks.py                ← Celery tasks
├── celery_app.py           ← Celery + Redis config
├── database.py             ← PostgreSQL + velocity checks
├── face_match.py           ← DeepFace liveness + matching
├── hologram.py             ← Security feature detection
├── template_match.py       ← Country ID template matching
├── requirements.txt
├── Dockerfile              ← Single image for API + all workers
├── docker-compose.yml      ← All 7 services wired together
├── .env                    ← Live config (edit before starting)
├── .env.example            ← Reference copy
├── .dockerignore
├── Makefile                ← Convenience shortcuts
└── nginx/
    ├── nginx.conf          ← Main Nginx config
    └── conf.d/
        └── fraud_api.conf  ← Virtual host + rate limiting
```

---

## Services at a glance

| Service | What it does | Port |
|---|---|---|
| `postgres` | Stores screening logs + velocity events | internal only |
| `redis` | Celery broker + result backend | internal only |
| `api` | FastAPI — receives uploads, submits tasks | internal only |
| `worker-pdf` | Processes PDF screening jobs | internal only |
| `worker-image` | Processes image screening jobs | internal only |
| `worker-idcard` | Processes ID card jobs (face, hologram, template) | internal only |
| `flower` | Celery monitoring dashboard | **5555** (direct) |
| `nginx` | Reverse proxy + rate limiting | **80** |

---

## First-time setup

```bash
# 1. Clone / copy all files into one directory
# 2. Edit credentials in .env if desired (defaults work for local dev)
# 3. Build the image (first build takes ~5 min — downloads DeepFace models)
make build

# 4. Start everything
make up

# 5. Check all services are healthy
make ps
```

Expected output of `make ps`:
```
NAME                STATUS          PORTS
fraud-postgres      running (healthy)
fraud-redis         running (healthy)
fraud-api           running (healthy)
fraud-worker-pdf    running
fraud-worker-image  running
fraud-worker-idcard running
fraud-flower        running         0.0.0.0:5555->5555/tcp
fraud-nginx         running         0.0.0.0:80->80/tcp
```

---

## Using the API

All requests go through Nginx on port 80:

```bash
# Submit an ID card + selfie
curl -X POST http://localhost/screen-id-card \
  -F "file=@id_card.jpg" \
  -F "selfie=@selfie.jpg" \
  -F "applicant_id=APP-001"

# → {"task_id": "abc123...", "status": "queued"}

# Poll for result
curl http://localhost/result/abc123...

# Submit a PDF
curl -X POST http://localhost/screen-pdf \
  -F "file=@bank_statement.pdf" \
  -F "applicant_id=APP-001"

# View OpenAPI docs
open http://localhost/docs
```

---

## Monitoring

| Dashboard | URL | Credentials |
|---|---|---|
| Flower (Celery) | http://localhost:5555 | admin / flowerpass |
| FastAPI docs | http://localhost/docs | none |

Flower shows: active workers, queued tasks, task history, failure rates.

---

## Common commands

```bash
make logs                    # tail all service logs
make log svc=worker-idcard   # tail one service
make restart svc=api         # restart just the API
make shell-api               # bash shell inside API container
make shell-db                # psql inside Postgres container
make clean                   # remove stopped containers + dangling images
make nuke                    # ⚠ full wipe including DB data
```

---

## Environment variables (.env)

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://fraud:fraudpass@postgres:5432/fraud_detect` | Uses service name `postgres` not `localhost` |
| `REDIS_URL` | `redis://redis:6379/0` | Uses service name `redis` |
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Linux path inside container — do not change |
| `UPLOAD_DIR` | `/app/uploads` | Shared Docker volume across all workers |
| `VELOCITY_WINDOW_HOURS` | `24` | Look-back window for velocity checks |
| `MAX_SUBMISSIONS_PER_ID` | `3` | Max same-ID submissions in window |
| `MAX_SAME_FILE_SUBMISSIONS` | `2` | Max exact duplicate files in window |
| `FLOWER_USER` | `admin` | Flower dashboard login |
| `FLOWER_PASSWORD` | `flowerpass` | Change before sharing access |

> **Important:** Inside Docker, always use service names (`postgres`, `redis`) not `localhost` in connection strings. This is already set correctly in the provided `.env`.

---

## Nginx rate limits

| Endpoint pattern | Limit |
|---|---|
| `/screen-*` | 10 req/s per IP, burst 5 |
| `/result/*`, `/history/*` | 30 req/s per IP, burst 20 |

Adjust in `nginx/conf.d/fraud_api.conf` → `limit_req_zone` directives.

---

## Troubleshooting

**API can't connect to Postgres on startup**
The API has `depends_on: postgres: condition: service_healthy` — it waits for the healthcheck to pass. If it still fails, run `make log svc=postgres` to check.

**DeepFace models downloading on first run**
DeepFace downloads ArcFace weights (~500 MB) on first use. The `worker-idcard` container will be slow on the very first ID card job. Subsequent jobs are fast. To pre-download during build, add this to the Dockerfile:
```dockerfile
RUN python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"
```

**Workers not picking up tasks**
Check `make log svc=worker-pdf` — confirm the worker connected to Redis and is listening on the correct queue.

**Upload too large (413 error)**
Increase `client_max_body_size` in both `nginx.conf` and `fraud_api.conf`.
