import os, re, json, logging
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
load_dotenv()
log = logging.getLogger("fraud_detect.network")

DATABASE_URL = os.getenv("DATABASE_URL","postgresql://postgres:password@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

PHONE_RE    = re.compile(r"(?:\+855|0)[\s\-]?(?:1[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9])[\s\-]?\d{3}[\s\-]?\d{3,4}")
EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
ACCT_RE     = re.compile(r"\b(?:account|acct\.?|a\/c)\s*(?:no\.?|number|#)?\s*:?\s*([0-9X\*]{6,20})\b", re.IGNORECASE)
EMPLOYER_RE = re.compile(r"(?:employer|company|paid by)\s*:?\s*([A-Z][A-Za-z\s\.\,\&]{3,40})", re.IGNORECASE)
ADDRESS_RE  = re.compile(r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl)[,\s]+[A-Za-z\s]+", re.IGNORECASE)

def _normalise_phone(p): return re.sub(r"[\s\-\(\)]","",p)
def _normalise_addr(a):  return re.sub(r"\s+"," ",a.lower().strip())

def _extract_network_attributes(ocr_text, result):
    attrs = defaultdict(list)
    if not ocr_text: return attrs
    for p in PHONE_RE.findall(ocr_text):   attrs["phone"].append(_normalise_phone(p))
    for e in EMAIL_RE.findall(ocr_text):   attrs["email"].append(e.lower().strip())
    for a in ACCT_RE.findall(ocr_text):
        c = re.sub(r"[X\*]","",a).strip()
        if len(c) >= 6: attrs["account"].append(c)
    for emp in EMPLOYER_RE.findall(ocr_text):
        c = emp.strip().upper()
        if len(c) >= 4: attrs["employer"].append(c)
    for addr in ADDRESS_RE.findall(ocr_text): attrs["address"].append(_normalise_addr(addr))
    fi = result.get("field_info",{})
    if fi.get("id_number"): attrs["id_number"].append(fi["id_number"])
    return dict(attrs)

def _init_network_table():
    db = SessionLocal()
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS fraud_network_links (
                id          SERIAL PRIMARY KEY, applicant_a VARCHAR(100),
                applicant_b VARCHAR(100), link_type VARCHAR(50),
                link_value  VARCHAR(500), strength INTEGER DEFAULT 1,
                first_seen  TIMESTAMP DEFAULT NOW(), last_seen TIMESTAMP DEFAULT NOW()
            )"""))
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS applicant_attributes (
                id           SERIAL PRIMARY KEY, applicant_id VARCHAR(100),
                attr_type    VARCHAR(50), attr_value VARCHAR(500),
                doc_type     VARCHAR(50), extracted_at TIMESTAMP DEFAULT NOW()
            )"""))
        db.execute(text("CREATE INDEX IF NOT EXISTS ix_appattr_value ON applicant_attributes(attr_type, attr_value)"))
        db.commit()
    except Exception as e:
        db.rollback(); log.error(f"Network table init: {e}")
    finally:
        db.close()

def _store_attributes(applicant_id, attrs, doc_type):
    if not applicant_id or not attrs: return
    db = SessionLocal()
    try:
        for attr_type, values in attrs.items():
            for val in values:
                if not val or len(val) < 3: continue
                db.execute(text("""
                    INSERT INTO applicant_attributes (applicant_id,attr_type,attr_value,doc_type)
                    VALUES (:aid,:t,:v,:dt) ON CONFLICT DO NOTHING
                """), {"aid":applicant_id,"t":attr_type,"v":val[:500],"dt":doc_type})
        db.commit()
    except Exception as e:
        db.rollback(); log.warning(f"Store attributes: {e}")
    finally:
        db.close()

def _find_shared_attributes(applicant_id, attrs):
    if not applicant_id or not attrs: return []
    db = SessionLocal(); matches = []
    try:
        for attr_type, values in attrs.items():
            for val in values:
                if not val or len(val) < 4: continue
                rows = db.execute(text("""
                    SELECT DISTINCT applicant_id, attr_type, attr_value, doc_type, extracted_at
                    FROM applicant_attributes WHERE attr_type=:t AND attr_value=:v
                    AND applicant_id!=:aid ORDER BY extracted_at DESC LIMIT 10
                """), {"t":attr_type,"v":val,"aid":applicant_id}).fetchall()
                for row in rows:
                    matches.append({"matched_applicant":row.applicant_id,"link_type":attr_type,
                                    "link_value":val,"doc_type":row.doc_type,
                                    "first_seen":str(row.extracted_at)[:16]})
    except Exception as e:
        log.warning(f"Find shared attributes: {e}")
    finally:
        db.close()
    return matches

def _store_links(applicant_id, matches):
    if not matches: return
    db = SessionLocal()
    try:
        for m in matches:
            db.execute(text("""
                INSERT INTO fraud_network_links (applicant_a,applicant_b,link_type,link_value,strength,last_seen)
                VALUES (:a,:b,:t,:v,1,NOW()) ON CONFLICT DO NOTHING
            """), {"a":applicant_id,"b":m["matched_applicant"],"t":m["link_type"],"v":m["link_value"][:500]})
        db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

def analyze_fraud_network(applicant_id, ocr_text, result, doc_type="id_card"):
    flags, info = [], {}
    if not applicant_id: return {"network_status":"no_applicant_id"}, []
    try:
        _init_network_table()
        attrs = _extract_network_attributes(ocr_text or "", result)
        info["attributes_extracted"] = {k:len(v) for k,v in attrs.items()}
        _store_attributes(applicant_id, attrs, doc_type)
        matches = _find_shared_attributes(applicant_id, attrs)
        _store_links(applicant_id, matches)
        if not matches:
            info["network_connections"] = 0
            return info, []
        by_applicant = defaultdict(list)
        for m in matches: by_applicant[m["matched_applicant"]].append(m)
        weight_map   = {"phone":5,"email":5,"account":4,"id_number":4,"address":3,"employer":2}
        connected    = []
        for other_id, links in by_applicant.items():
            link_types = list(set(l["link_type"] for l in links))
            ws = sum(weight_map.get(lt,1) for lt in link_types)
            connected.append({"applicant_id":other_id,"link_count":len(links),
                               "link_types":link_types,"weighted_score":ws,"links":links[:5]})
        connected.sort(key=lambda x: x["weighted_score"], reverse=True)
        info["network_connections"]  = len(connected)
        info["connected_applicants"] = connected[:10]
        info["attributes_found"]     = attrs
        if len(connected) >= 5:    flags.append("fraud_ring_suspected")
        elif len(connected) >= 2:  flags.append("network_connections_detected")
        elif connected:            flags.append("single_network_connection")
        if any("phone"   in c["link_types"] for c in connected): flags.append("shared_phone_number")
        if any("email"   in c["link_types"] for c in connected): flags.append("shared_email_address")
        if any("account" in c["link_types"] for c in connected): flags.append("shared_bank_account")
        if any(c["link_count"] >= 2 for c in connected):         flags.append("multiple_shared_attributes")
        info["network_risk_score"] = sum(c["weighted_score"] for c in connected)
    except Exception as e:
        log.error(f"Network analysis failed: {e}")
        info["network_error"] = str(e)
    return info, flags

def get_network_graph(applicant_id=None, days=90):
    since = datetime.utcnow() - timedelta(days=days)
    db    = SessionLocal()
    try:
        if applicant_id:
            rows = db.execute(text("""
                SELECT applicant_a,applicant_b,link_type,link_value,strength
                FROM fraud_network_links WHERE (applicant_a=:aid OR applicant_b=:aid)
                ORDER BY strength DESC LIMIT 50
            """), {"aid":applicant_id}).fetchall()
        else:
            rows = db.execute(text("""
                SELECT applicant_a,applicant_b,link_type,link_value,strength
                FROM fraud_network_links WHERE first_seen>=:since
                ORDER BY strength DESC LIMIT 100
            """), {"since":since}).fetchall()
        nodes = set(); edges = []
        for r in rows:
            nodes.add(r.applicant_a); nodes.add(r.applicant_b)
            lv = r.link_value
            edges.append({"source":r.applicant_a,"target":r.applicant_b,
                          "link_type":r.link_type,
                          "link_value":lv[:30]+"…" if len(lv or "")>30 else lv,
                          "strength":r.strength})
        return {"nodes":[{"id":n} for n in nodes],"edges":edges,
                "node_count":len(nodes),"edge_count":len(edges)}
    except Exception as e:
        return {"nodes":[],"edges":[],"error":str(e)}
    finally:
        db.close()

def get_network_stats(days=30):
    since = datetime.utcnow() - timedelta(days=days)
    db    = SessionLocal()
    try:
        total = db.execute(text("SELECT COUNT(*) FROM fraud_network_links WHERE first_seen>=:s"),{"s":since}).scalar() or 0
        rings = db.execute(text("""
            SELECT applicant_a, COUNT(DISTINCT applicant_b) c FROM fraud_network_links
            WHERE first_seen>=:s GROUP BY applicant_a HAVING COUNT(DISTINCT applicant_b)>=3
        """),{"s":since}).fetchall()
        by_type = db.execute(text("""
            SELECT link_type, COUNT(*) cnt FROM fraud_network_links
            WHERE first_seen>=:s GROUP BY link_type ORDER BY cnt DESC
        """),{"s":since}).fetchall()
        return {"total_links":total,"suspected_rings":len(rings),
                "links_by_type":{r.link_type:r.cnt for r in by_type},"period_days":days}
    except Exception as e:
        return {"error":str(e)}
    finally:
        db.close()