"""
loan_integration.py — Loan Case Integration API
Single endpoint that accepts all loan documents in one call.
Designed for integration with external loan origination systems.
"""
import os
import uuid
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
log = logging.getLogger("fraud_detect.loan")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud:fraudpass@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="KYC Fraud Detection — Loan Integration API",
    description="""
## Loan Case Integration API

Submit all loan application documents in a single API call.
Receive a unified risk assessment with per-document breakdown.

### Authentication
```
X-API-Key: fds_your_api_key_here
```

### Quick Start
```bash
curl -X POST http://172.16.26.48:3000/api/v1/loan-case \\
  -H "X-API-Key: fds_your_key" \\
  -F "loan_ref=LOAN-2024-001" \\
  -F "applicant_id=APP-001" \\
  -F "applicant_name=John Smith" \\
  -F "id_card=@id_card.jpg" \\
  -F "bank_statement=@statement.pdf" \\
  -F "callback_url=https://your-system/webhook/kyc"
```
    """,
    version="1.0.0",
)

API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"


def _verify_key(key: str) -> bool:
    if not key:
        return False
    h  = hashlib.sha256(key.encode()).hexdigest()
    db = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT id FROM api_keys WHERE key_hash=:h AND active=TRUE"
        ), {"h": h}).fetchone()
        if row:
            db.execute(text("UPDATE api_keys SET last_used=NOW() WHERE id=:id"), {"id": row.id})
            db.commit()
            return True
        return False
    except:
        return True
    finally:
        db.close()


def _auth(request: Request):
    if not API_AUTH_ENABLED:
        return
    key = (request.headers.get("X-API-Key") or
           request.headers.get("Authorization", "").replace("Bearer ", ""))
    if not _verify_key(key):
        raise HTTPException(401, detail={
            "error": "invalid_api_key",
            "message": "Include header: X-API-Key: fds_your_key",
        })


def _aggregate_risk(doc_results: list) -> tuple:
    levels = [d.get("risk_level", "LOW") for d in doc_results]
    scores = [d.get("risk_score", 0) for d in doc_results]
    max_score = max(scores) if scores else 0

    if "HIGH" in levels:
        return "HIGH", max_score, "REJECT"
    if levels.count("MEDIUM") >= 2:
        return "HIGH", max_score, "REJECT"
    if "MEDIUM" in levels:
        return "MEDIUM", max_score, "REVIEW"
    return "LOW", max_score, "PASS"


def _fire_webhook(url: str, payload: dict):
    if not url:
        return
    import urllib.request
    try:
        body = json.dumps(payload, default=str).encode()
        req  = urllib.request.Request(url, data=body, method="POST",
               headers={"Content-Type": "application/json",
                        "X-Event-Type": "loan_case.complete"})
        with urllib.request.urlopen(req, timeout=10) as r:
            log.info(f"Webhook → {url} HTTP {r.status}")
    except Exception as e:
        log.warning(f"Webhook failed: {e}")


def _init_loan_table():
    db = SessionLocal()
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS loan_cases (
                id              SERIAL PRIMARY KEY,
                loan_ref        VARCHAR(100) UNIQUE,
                applicant_id    VARCHAR(100),
                applicant_name  VARCHAR(200),
                overall_risk    VARCHAR(10),
                overall_score   INTEGER,
                overall_action  VARCHAR(20),
                document_count  INTEGER,
                results         JSONB,
                callback_url    VARCHAR(500),
                status          VARCHAR(20) DEFAULT 'complete',
                created_at      TIMESTAMP DEFAULT NOW(),
                completed_at    TIMESTAMP
            )
        """))
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"loan table init: {e}")
    finally:
        db.close()


def _save_loan_case(data: dict):
    db = SessionLocal()
    try:
        db.execute(text("""
            INSERT INTO loan_cases
            (loan_ref, applicant_id, applicant_name, overall_risk, overall_score,
             overall_action, document_count, results, callback_url, status, completed_at)
            VALUES (:ref, :aid, :name, :risk, :score, :action, :docs,
                    :results::jsonb, :cb, 'complete', NOW())
            ON CONFLICT (loan_ref) DO UPDATE SET
                overall_risk=EXCLUDED.overall_risk,
                overall_score=EXCLUDED.overall_score,
                overall_action=EXCLUDED.overall_action,
                results=EXCLUDED.results,
                completed_at=NOW()
        """), {
            "ref":     data.get("loan_ref"),
            "aid":     data.get("applicant_id"),
            "name":    data.get("applicant_name"),
            "risk":    data.get("overall_risk"),
            "score":   data.get("overall_score"),
            "action":  data.get("overall_action"),
            "docs":    data.get("document_count"),
            "results": json.dumps(data.get("documents", {})),
            "cb":      data.get("callback_url"),
        })
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"save_loan_case: {e}")
    finally:
        db.close()


def _get_loan_case(loan_ref: str):
    db = SessionLocal()
    try:
        row = db.execute(text("SELECT * FROM loan_cases WHERE loan_ref=:ref"),
                         {"ref": loan_ref}).fetchone()
        if not row:
            return None
        r = dict(row._mapping)
        for k in ["created_at", "completed_at"]:
            if r.get(k): r[k] = r[k].isoformat()
        return r
    finally:
        db.close()


def _list_loan_cases(days: int = 30, limit: int = 50):
    since = datetime.utcnow() - timedelta(days=days)
    db    = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT loan_ref, applicant_id, applicant_name, overall_risk,
                   overall_score, overall_action, document_count, status, created_at
            FROM loan_cases WHERE created_at >= :since
            ORDER BY created_at DESC LIMIT :limit
        """), {"since": since, "limit": limit}).fetchall()
        result = []
        for r in rows:
            rd = dict(r._mapping)
            if rd.get("created_at"): rd["created_at"] = rd["created_at"].isoformat()
            result.append(rd)
        return result
    finally:
        db.close()


@app.on_event("startup")
def startup():
    _init_loan_table()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOAN CASE ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/loan-case", summary="Submit all loan documents for fraud screening")
async def submit_loan_case(
    request:        Request,
    loan_ref:       str = Form(...,  description="Unique loan reference number"),
    applicant_id:   str = Form(...,  description="Applicant/customer ID"),
    applicant_name: str = Form("",   description="Applicant full name"),
    loan_type:      str = Form("",   description="e.g. Personal, Mortgage, SME"),
    loan_amount:    str = Form("",   description="Requested loan amount"),
    id_card:        UploadFile | None = File(None, description="National ID or passport image"),
    selfie:         UploadFile | None = File(None, description="Applicant selfie for face match"),
    bank_statement: UploadFile | None = File(None, description="Bank statement PDF"),
    payslip:        UploadFile | None = File(None, description="Payslip PDF"),
    utility_bill:   UploadFile | None = File(None, description="Utility bill"),
    other_doc:      UploadFile | None = File(None, description="Other supporting document"),
    callback_url:   str = Form("",   description="Webhook URL for result notification"),
    source_system:  str = Form("",   description="Name of calling system"),
):
    _auth(request)

    docs_provided = [f for f in [id_card, bank_statement, payslip, utility_bill, other_doc]
                     if f and f.filename]
    if not docs_provided:
        raise HTTPException(400, {"error": "no_documents",
                                  "message": "At least one document must be provided"})

    import httpx
    import asyncio

    started_at  = datetime.utcnow().isoformat() + "Z"
    doc_results = {}
    all_flags   = []

    async with httpx.AsyncClient(base_url="http://api:8001", timeout=120) as client:

        # ── ID Card ───────────────────────────────────────────────────────────
        if id_card and id_card.filename:
            try:
                files = {"file": (id_card.filename, await id_card.read(),
                                  id_card.content_type or "image/jpeg")}
                if selfie and selfie.filename:
                    files["selfie"] = (selfie.filename, await selfie.read(),
                                       selfie.content_type or "image/jpeg")
                r = await client.post("/screen-id-card", files=files,
                                      data={"applicant_id": applicant_id})
                if r.status_code == 202:
                    task_id = r.json().get("task_id")
                    result  = None
                    for _ in range(12):
                        await asyncio.sleep(5)
                        poll = await client.get(f"/result/{task_id}")
                        pd   = poll.json()
                        if pd.get("status") == "complete":
                            result = pd.get("result", {})
                            break
                        elif pd.get("status") == "failed":
                            break
                    if result:
                        risk = result.get("risk", {})
                        doc_results["id_card"] = {
                            "document":   id_card.filename,
                            "task_id":    task_id,
                            "risk_level": risk.get("level", "UNKNOWN"),
                            "risk_score": risk.get("score", 0),
                            "action":     risk.get("action", "REVIEW"),
                            "flags":      result.get("flags", []),
                            "field_info": result.get("field_info", {}),
                            "face_match": result.get("face_match", {}),
                            "ml":         result.get("ml_inference", {}),
                        }
                        all_flags.extend(result.get("flags", []))
                    else:
                        doc_results["id_card"] = {
                            "document": id_card.filename,
                            "risk_level": "MEDIUM",
                            "flags": ["id_card_screening_timeout"],
                        }
            except Exception as e:
                log.error(f"ID card error: {e}")
                doc_results["id_card"] = {"error": str(e), "risk_level": "MEDIUM"}

        # ── PDF documents ─────────────────────────────────────────────────────
        pdf_docs = {
            "bank_statement": bank_statement,
            "payslip":        payslip,
        }
        for doc_key, doc_file in pdf_docs.items():
            if not doc_file or not doc_file.filename:
                continue
            try:
                ext = Path(doc_file.filename).suffix.lower()
                endpoint = "/screen-pdf" if ext == ".pdf" else "/screen-image"
                ct = "application/pdf" if ext == ".pdf" else (doc_file.content_type or "image/jpeg")
                r  = await client.post(endpoint,
                     files={"file": (doc_file.filename, await doc_file.read(), ct)},
                     data={"applicant_id": applicant_id})
                if r.status_code == 200:
                    res  = r.json()
                    risk = res.get("risk", {})
                    doc_results[doc_key] = {
                        "document":     doc_file.filename,
                        "risk_level":   risk.get("level") or res.get("risk_level", "UNKNOWN"),
                        "risk_score":   risk.get("score") or res.get("risk_score", 0),
                        "action":       risk.get("action", "REVIEW"),
                        "flags":        res.get("flags", []),
                        "field_checks": res.get("field_checks", {}),
                    }
                    all_flags.extend(res.get("flags", []))
            except Exception as e:
                log.error(f"{doc_key} error: {e}")
                doc_results[doc_key] = {"error": str(e), "risk_level": "MEDIUM"}

        # ── Other docs ────────────────────────────────────────────────────────
        other_docs = {
            "utility_bill": utility_bill,
            "other_doc":    other_doc,
        }
        for doc_key, doc_file in other_docs.items():
            if not doc_file or not doc_file.filename:
                continue
            try:
                ext = Path(doc_file.filename).suffix.lower()
                endpoint = "/screen-pdf" if ext == ".pdf" else "/screen-image"
                ct = "application/pdf" if ext == ".pdf" else (doc_file.content_type or "image/jpeg")
                r  = await client.post(endpoint,
                     files={"file": (doc_file.filename, await doc_file.read(), ct)},
                     data={"applicant_id": applicant_id})
                if r.status_code == 200:
                    res  = r.json()
                    risk = res.get("risk", {})
                    doc_results[doc_key] = {
                        "document":   doc_file.filename,
                        "risk_level": risk.get("level") or res.get("risk_level", "UNKNOWN"),
                        "risk_score": risk.get("score") or res.get("risk_score", 0),
                        "action":     risk.get("action", "REVIEW"),
                        "flags":      res.get("flags", []),
                    }
                    all_flags.extend(res.get("flags", []))
            except Exception as e:
                log.error(f"{doc_key} error: {e}")
                doc_results[doc_key] = {"error": str(e), "risk_level": "MEDIUM"}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    doc_list   = list(doc_results.values())
    o_risk, o_score, o_action = _aggregate_risk(doc_list)
    unique_flags = list(dict.fromkeys(all_flags))

    response = {
        "loan_ref":       loan_ref,
        "applicant_id":   applicant_id,
        "applicant_name": applicant_name,
        "loan_type":      loan_type,
        "loan_amount":    loan_amount,
        "source_system":  source_system,
        "screened_at":    started_at,
        "completed_at":   datetime.utcnow().isoformat() + "Z",

        "overall": {
            "risk_level":    o_risk,
            "risk_score":    o_score,
            "action":        o_action,
            "document_count": len(doc_list),
            "flag_count":    len(unique_flags),
            "description": {
                "PASS":   "All documents appear authentic. Proceed with loan processing.",
                "REVIEW": "Some documents require manual verification before proceeding.",
                "REJECT": "Significant fraud indicators detected. Do not proceed without investigation.",
            }.get(o_action, ""),
            "flags": unique_flags[:20],
        },

        "documents":  doc_results,

        "integration": {
            "api_version":  "1.0",
            "callback_url": callback_url or None,
        }
    }

    _save_loan_case({
        "loan_ref": loan_ref, "applicant_id": applicant_id,
        "applicant_name": applicant_name,
        "overall_risk": o_risk, "overall_score": o_score,
        "overall_action": o_action, "document_count": len(doc_list),
        "documents": doc_results, "callback_url": callback_url,
    })

    if callback_url:
        _fire_webhook(callback_url, {
            "event": "loan_case.complete",
            "loan_ref": loan_ref,
            "applicant_id": applicant_id,
            "overall_risk": o_risk,
            "overall_action": o_action,
            "detail": response,
        })

    return JSONResponse(response)


@app.get("/loan-case/{loan_ref}", summary="Get result for a loan case")
async def get_loan_case(loan_ref: str, request: Request):
    _auth(request)
    case = _get_loan_case(loan_ref)
    if not case:
        raise HTTPException(404, {"error": "not_found", "loan_ref": loan_ref})
    return JSONResponse(case)


@app.get("/loan-cases", summary="List recent loan cases")
async def list_loan_cases(request: Request, days: int = 30, limit: int = 50):
    _auth(request)
    return JSONResponse({"cases": _list_loan_cases(days, limit)})


@app.get("/", summary="API info")
async def api_info():
    return JSONResponse({
        "service": "KYC Fraud Detection — Loan Integration API",
        "version": "1.0.0",
        "status":  "running",
        "auth":    "required" if API_AUTH_ENABLED else "disabled",
        "endpoints": {
            "POST /api/v1/loan-case":       "Submit all loan documents — unified risk assessment",
            "GET  /api/v1/loan-case/{ref}": "Get result by loan reference",
            "GET  /api/v1/loan-cases":      "List recent loan cases",
        },
        "supported_documents": {
            "id_card":        "National ID/passport — full forensic + face match",
            "selfie":         "Selfie for face matching against ID",
            "bank_statement": "Bank statement PDF",
            "payslip":        "Payslip PDF or image",
            "utility_bill":   "Utility bill PDF or image",
            "other_doc":      "Any other document",
        },
        "risk_aggregation": {
            "any HIGH":  "Overall = HIGH → REJECT",
            "2+ MEDIUM": "Overall = HIGH → REJECT",
            "1 MEDIUM":  "Overall = MEDIUM → REVIEW",
            "all LOW":   "Overall = LOW → PASS",
        },
        "docs": "/api/v1/docs",
    })
