"""
cbc_api.py — CBC API Client + Enquiry Cache Manager

Cache: JSON file store in /app/uploads/cbc/api_cache/
Key:   NID number
TTL:   30 days (configurable)
Force refresh: admin only
"""
import os, json, hashlib, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import requests

log = logging.getLogger("cbc_api")

# ── Config ────────────────────────────────────────────────────────────────────
API_URL     = os.getenv("CBC_API_URL", "https://api.cbc.com.kh/ws/service")
API_USER    = os.getenv("CBC_API_USER", "")
API_MEMBER  = os.getenv("CBC_API_MEMBER_ID", "")
API_TIMEOUT = int(os.getenv("CBC_API_TIMEOUT", "30"))
CACHE_TTL   = int(os.getenv("CBC_CACHE_DAYS", "30"))

UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
CACHE_DIR   = UPLOAD_DIR / "cbc" / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_key(nid: str) -> str:
    return hashlib.md5(nid.strip().upper().encode()).hexdigest()

def _cache_path(nid: str) -> Path:
    return CACHE_DIR / (_cache_key(nid) + ".json")

def cache_get(nid: str) -> Optional[dict]:
    """Return cached result if exists and not expired."""
    p = _cache_path(nid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        if datetime.now() - cached_at > timedelta(days=CACHE_TTL):
            return None  # expired
        return data
    except:
        return None

def cache_set(nid: str, result: dict, username: str = "system"):
    """Save result to cache."""
    p = _cache_path(nid)
    result["cached_at"]  = datetime.now().isoformat()
    result["cached_by"]  = username
    result["cache_nid"]  = nid.strip().upper()
    p.write_text(json.dumps(result, indent=2, default=str))

def cache_delete(nid: str):
    """Delete cached result (admin force refresh)."""
    p = _cache_path(nid)
    if p.exists():
        p.unlink()

def cache_info(nid: str) -> Optional[dict]:
    """Return cache metadata without full result."""
    p = _cache_path(nid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        cached_at = datetime.fromisoformat(data.get("cached_at","2000-01-01"))
        expires   = cached_at + timedelta(days=CACHE_TTL)
        expired   = datetime.now() > expires
        return {
            "nid":        data.get("cache_nid",""),
            "cached_at":  cached_at.strftime("%Y-%m-%d %H:%M"),
            "cached_by":  data.get("cached_by",""),
            "expires":    expires.strftime("%Y-%m-%d"),
            "expired":    expired,
            "days_left":  max(0, (expires - datetime.now()).days),
        }
    except:
        return None

def list_cache() -> list:
    """List all cached enquiries."""
    results = []
    for p in sorted(CACHE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            cached_at = datetime.fromisoformat(data.get("cached_at","2000-01-01"))
            expires   = cached_at + timedelta(days=CACHE_TTL)
            results.append({
                "nid":       data.get("cache_nid",""),
                "name":      _get_name(data),
                "cached_at": cached_at.strftime("%Y-%m-%d %H:%M"),
                "cached_by": data.get("cached_by",""),
                "expires":   expires.strftime("%Y-%m-%d"),
                "expired":   datetime.now() > expires,
                "days_left": max(0, (expires - datetime.now()).days),
            })
        except: pass
    return results

def _get_name(data: dict) -> str:
    """Extract name from cached result."""
    try:
        apps = data.get("applicants",[])
        if apps:
            p = apps[0].get("personal",{})
            return p.get("full_name_en", p.get("name_en_family",""))
    except: pass
    return ""

# ── API Client ────────────────────────────────────────────────────────────────

def build_request(
    nid: str,
    amount: str,
    currency: str,
    enquiry_ref: str,
    # Optional fields
    name_family: str = "",
    name_first:  str = "",
    dob_d: str = "", dob_m: str = "", dob_y: str = "",
    gender: str = "M",
    product_type: str = "GRL",
    account_type: str = "S",
) -> dict:
    """Build CBC API request payload."""
    return {
        "REQUEST": {
            "SERVICE":   "CBAJSNEN2",
            "ACTION":    "A_SC",
            "USER":      API_USER,
            "MEMBER_ID": API_MEMBER,
            "MESSAGE": {
                "ENQUIRY": {
                    "ENQUIRY_TYPE":        "NA",
                    "PRODUCT_TYPE":        product_type,
                    "NO_OF_APPLICANTS":    "1",
                    "ACCOUNT_TYPE":        account_type,
                    "ENQUIRY_REFERENCE":   enquiry_ref,
                    "AMOUNT":              str(amount),
                    "CURRENCY":            currency.upper(),
                    "CONSUMERS": [{
                        "CAPL": "P",
                        "CIDS": [{
                            "CID1": "N",
                            "CID2": nid.strip(),
                            "CID3": {"CID3D": "", "CID3M": "", "CID3Y": ""}
                        }],
                        "CDOB": {
                            "CDBD": dob_d,
                            "CDBM": dob_m,
                            "CDBY": dob_y
                        },
                        "CGND": gender,
                        "CMAR": "",
                        "CNAT": "KHM",
                        "CPLB": {"CPLBC": "KHM", "CPLBP": "", "CPLBD": "", "CPLBCM": ""},
                        "CNAMS": [{
                            "CNMFA": "",
                            "CNM1A": "",
                            "CNMFE": name_family,
                            "CNM1E": name_first,
                            "CNM2E": "",
                            "CNM3E": ""
                        }],
                        "CEML":  "",
                        "CADRS": [],
                        "CCNTS": [],
                        "CEMPS": []
                    }]
                }
            }
        }
    }


def call_api(payload: dict) -> dict:
    """Call CBC API and return response dict."""
    if not API_URL or not API_USER or not API_MEMBER:
        raise ValueError("CBC API not configured. Set CBC_API_URL, CBC_API_USER, CBC_API_MEMBER_ID in .env")

    headers = {"Content-Type": "application/json"}
    resp = requests.post(
        API_URL,
        json=payload,
        headers=headers,
        timeout=API_TIMEOUT,
        verify=True,
    )
    resp.raise_for_status()
    return resp.json()


def parse_api_response(raw: dict) -> dict:
    """
    Parse CBC API JSON response into the same structure
    as cbc_extractor.py produces from PDF parsing.
    Maps API fields to internal format so existing result page works.
    """
    # TODO: Map actual API response fields once sample response is provided
    # Placeholder structure matching cbc_extractor output
    result = {
        "source":     "api",
        "header":     {},
        "applicants": [],
    }

    try:
        resp  = raw.get("RESPONSE", raw)
        msg   = resp.get("MESSAGE", {})
        enq   = msg.get("ENQUIRY", {})

        # Header
        result["header"] = {
            "report_date":    enq.get("REPORT_DATE",""),
            "enquiry_number": enq.get("ENQUIRY_REFERENCE",""),
            "enquiry_type":   enq.get("ENQUIRY_TYPE",""),
            "product_type":   enq.get("PRODUCT_TYPE",""),
            "num_applicants": enq.get("NO_OF_APPLICANTS","1"),
            "account_type":   enq.get("ACCOUNT_TYPE",""),
            "amount":         enq.get("AMOUNT",""),
            "currency":       enq.get("CURRENCY",""),
        }

        # Consumers/Applicants
        consumers = enq.get("CONSUMERS", [])
        for consumer in consumers:
            personal  = _parse_consumer_personal(consumer)
            summary   = _parse_consumer_summary(consumer)
            accounts  = _parse_consumer_accounts(consumer)
            result["applicants"].append({
                "personal":  personal,
                "summary":   summary,
                "accounts":  accounts,
                "is_primary": consumer.get("CAPL","P") == "P",
            })

    except Exception as e:
        log.error(f"API response parse error: {e}")
        result["parse_error"] = str(e)

    return result


def _parse_consumer_personal(c: dict) -> dict:
    """Map consumer fields to personal info dict."""
    name = (c.get("CNAMS") or [{}])[0]
    cid  = (c.get("CIDS")  or [{}])[0]
    dob  = c.get("CDOB", {})

    family = name.get("CNMFE","")
    first  = name.get("CNM1E","")

    return {
        "applicant_type":  c.get("CAPL","P"),
        "id_type":         "National ID",
        "id_number":       cid.get("CID2",""),
        "name_en_family":  family,
        "name_en_first":   first,
        "full_name_en":    f"{family} {first}".strip(),
        "dob":             f"{dob.get('CDBD','')}/{dob.get('CDBM','')}/{dob.get('CDBY','')}",
        "gender":          "Male" if c.get("CGND","") == "M" else "Female",
        "nationality":     c.get("CNAT",""),
    }


def _parse_consumer_summary(c: dict) -> dict:
    """Map consumer summary fields — fill once actual response is known."""
    # TODO: Map from actual API response fields
    summ = c.get("SUMMARY", c.get("CSUMM", {}))
    return {
        "total_accounts":    summ.get("TOTAL_ACCOUNTS", ""),
        "normal_accounts":   summ.get("NORMAL_ACCOUNTS", ""),
        "closed_accounts":   summ.get("CLOSED_ACCOUNTS", ""),
        "delinquent_accounts": summ.get("DELINQUENT_ACCOUNTS", ""),
        "total_limits":      summ.get("TOTAL_LIMITS", ""),
        "total_liabilities": summ.get("TOTAL_LIABILITIES", ""),
        "credit_score":      summ.get("SCORE", ""),
    }


def _parse_consumer_accounts(c: dict) -> dict:
    """Map account/loan records — fill once actual response is known."""
    # TODO: Map from actual API response fields
    accounts = []
    raw_accounts = c.get("ACCOUNTS", c.get("CACCTS", []))
    for acc in raw_accounts:
        accounts.append({
            "lender":       acc.get("CREDITOR_NAME", acc.get("CNAME","")),
            "loan_type":    acc.get("PRODUCT_TYPE", acc.get("CPTYP","")),
            "loan_amount":  acc.get("LIMIT_AMOUNT", acc.get("CLAMT",0)),
            "outstanding":  acc.get("OUTSTANDING",  acc.get("COSAM",0)),
            "currency":     acc.get("CURRENCY",     acc.get("CCURR","USD")),
            "status":       acc.get("ACCOUNT_STATUS",acc.get("CSTAT","Normal")),
            "open_date":    acc.get("OPEN_DATE",    acc.get("COPDT","")),
        })
    return {
        "accounts":          [a for a in accounts if a.get("status","").lower() != "closed"],
        "closed_accounts":   [a for a in accounts if a.get("status","").lower() == "closed"],
        "writeoff_accounts": [],
        "guaranteed_active": [],
        "guaranteed_closed": [],
    }


def enquire(
    nid: str,
    amount: str,
    currency: str,
    username: str = "system",
    force: bool = False,
    **kwargs
) -> tuple[dict, bool]:
    """
    Main entry point.
    Returns (result_dict, from_cache: bool)
    Raises ValueError if API not configured or call fails.
    """
    # Check cache first
    if not force:
        cached = cache_get(nid)
        if cached:
            log.info(f"Cache hit for NID {nid}")
            return cached, True

    # Generate unique ref
    import uuid
    ref = f"CAT_{nid}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    payload  = build_request(nid, amount, currency, ref, **kwargs)
    raw_resp = call_api(payload)
    result   = parse_api_response(raw_resp)

    # Store raw response for debugging
    result["_raw_response"] = raw_resp
    result["enquiry_ref"]   = ref
    result["enquiry_date"]  = datetime.now().isoformat()
    result["enquired_by"]   = username
    result["nid"]           = nid
    result["amount"]        = amount
    result["currency"]      = currency

    # Cache it
    cache_set(nid, result, username)
    log.info(f"API enquiry complete for NID {nid}, ref={ref}")

    return result, False
