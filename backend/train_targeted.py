"""
train_targeted.py — Targeted Training for Large Document Collections

Designed for 65,000+ mixed document datasets.
Filters by document type, processes in batches with progress,
shows estimated time remaining.

Usage:
  python train_targeted.py scan     -- analyse what's in the folder
  python train_targeted.py images   -- process ID card images only (~1000)
  python train_targeted.py pdfs     -- process banking PDFs only (CBC, salary, bank)
  python train_targeted.py all      -- process everything relevant
  python train_targeted.py train    -- train ML model after processing
  python train_targeted.py report   -- show results
"""

import os, sys, json, re, time, hashlib, logging, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
logging.basicConfig(
    level   = logging.WARNING,
    format  = '%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger("train_targeted")

DATABASE_URL = os.getenv("DATABASE_URL",
    "postgresql://postgres:password@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=3)
SessionLocal = sessionmaker(bind=engine)

DATA_DIR = Path("/DATA/all_documents")
ML_DIR   = Path("ml_data")
ML_DIR.mkdir(exist_ok=True)

# ── Document type detection by filename ──────────────────────────────────────

KEYWORD_MAP = {
    "bank_statement": [
        "statement","bank","acleda","aba","caminco","wing",
        "canadia","maybank","lolc","hattha","amret","prasac"
    ],
    "cbc_report":  ["cbc","credit bureau","credit report","crr"],
    "payslip":     ["salary","payslip","pay slip","wage","payroll","income"],
    "nid":         ["nid","national id","nationalid","khmer id"],
    "invoice":     ["invoice","invioce","receipt"],
    "contract":    ["contract","agreement","rental","lease"],
    "valuation":   ["valuation","appraisal","assessment"],
    "loan_form":   ["loan application","lrf","loan form","checklist"],
}

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".jfif",".JPG",".PNG",".JPEG"}
PDF_EXTS   = {".pdf",".PDF"}

def _detect_type(filepath: Path) -> str:
    name = filepath.name.lower()
    ext  = filepath.suffix

    if ext in IMAGE_EXTS:
        if any(k in name for k in ["nid","national","id card","khmer","passport"]): return "id_card"
        return "image"

    if ext in PDF_EXTS:
        for dtype, keywords in KEYWORD_MAP.items():
            if any(k in name for k in keywords): return dtype
        return "pdf_other"

    return "other"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


# ── DB setup ──────────────────────────────────────────────────────────────────

def _init_db():
    db = SessionLocal()
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS training_documents (
                id           SERIAL PRIMARY KEY,
                file_path    TEXT UNIQUE,
                file_name    VARCHAR(500),
                file_hash    VARCHAR(64),
                doc_type     VARCHAR(50),
                risk_score   INTEGER DEFAULT 0,
                risk_level   VARCHAR(10) DEFAULT 'LOW',
                flags        JSONB,
                full_result  JSONB,
                label        VARCHAR(20),
                label_source VARCHAR(30),
                ingested_at  TIMESTAMP DEFAULT NOW(),
                notes        TEXT
            )
        """))
        db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()


def _q(sql, params={}):
    db = SessionLocal()
    try:
        return [dict(r._mapping) for r in db.execute(text(sql), params)]
    except Exception as e:
        return []
    finally:
        db.close()


def _exec(sql, params={}):
    db = SessionLocal()
    try:
        db.execute(text(sql), params); db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()


def _already_ingested() -> set:
    rows = _q("SELECT file_path FROM training_documents WHERE file_path IS NOT NULL")
    return set(r["file_path"] for r in rows)


# ── Progress display ───────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int, label: str = ""):
        self.total   = total
        self.current = 0
        self.label   = label
        self.start   = time.time()
        self.ok      = 0
        self.err     = 0

    def update(self, success: bool = True, msg: str = ""):
        self.current += 1
        if success: self.ok  += 1
        else:       self.err += 1

        elapsed = time.time() - self.start
        rate    = self.current / max(elapsed, 0.001)
        remain  = (self.total - self.current) / max(rate, 0.001)
        pct     = self.current / max(self.total, 1) * 100
        bar_len = 30
        filled  = int(bar_len * pct / 100)
        bar     = "█" * filled + "░" * (bar_len - filled)

        eta = str(timedelta(seconds=int(remain)))
        print(f"\r[{bar}] {pct:5.1f}% {self.current}/{self.total} "
              f"ETA:{eta} OK:{self.ok} ERR:{self.err}  {msg[:30]:<30}",
              end="", flush=True)

    def done(self):
        elapsed = time.time() - self.start
        print(f"\n\nCompleted in {timedelta(seconds=int(elapsed))} — "
              f"{self.ok} processed, {self.err} errors")


# ── Screening functions ────────────────────────────────────────────────────────

def _screen_image(file_path: str) -> dict:
    """Screen ID card / image through KYC engine."""
    import cv2
    from screening import (
        perform_ela, noise_analysis, analyze_id_card_regions,
        copy_move_detection, basic_image_forensics,
        extract_id_card_fields, auto_rotate_id_card,
        score_id_card, score_image
    )
    img = cv2.imread(file_path)
    if img is None:
        return {"flags":[], "risk":{"level":"LOW","score":0}, "error":"unreadable"}

    # Auto-rotate
    try: img = auto_rotate_id_card(img)
    except: pass

    try:
        ela    = perform_ela(img)
        noise  = noise_analysis(img)
        region = analyze_id_card_regions(img)
        cm     = copy_move_detection(img)
        forensics = basic_image_forensics(img, file_path)
    except Exception as e:
        return {"flags":[],"risk":{"level":"LOW","score":0},"error":str(e)}

    all_flags = []
    for d in [ela, noise, region, cm, forensics]:
        if isinstance(d, dict):
            all_flags.extend(d.get("flags", []))

    # Try id_card scoring first, fall back to image scoring
    try:
        score, level = score_id_card(all_flags)
    except Exception:
        score, level = score_image(all_flags)

    return {
        "flags":    all_flags,
        "ela":      ela,
        "noise_analysis": noise,
        "region_info":    region,
        "copy_move":      cm,
        "forensics":      forensics,
        "risk": {"level": level, "score": score},
    }


def _screen_pdf(file_path: str, doc_type: str) -> dict:
    """Screen PDF through banking intelligence."""
    try:
        from pdf_banking import analyze_banking_pdf
        info, flags, score, level = analyze_banking_pdf(
            file_path, "", 0, doc_type)
        return {
            "flags": flags,
            "banking_analysis": info,
            "risk": {"level": level, "score": score},
        }
    except Exception as e:
        return {"flags":[],"risk":{"level":"LOW","score":0},"error":str(e)}


def _auto_label(risk_score: int, flags: list,
                high_thresh: int, low_thresh: int) -> str:
    definite = {
        "mrz_checksum_failed","face_mismatch",
        "balance_math_inconsistency","benfords_law_violation",
        "template_shared_across_applicants","fraud_ring_suspected",
        "future_dates_detected","no_tax_deductions_detected",
    }
    if any(f in flags for f in definite): return "tampered"
    if risk_score >= high_thresh:         return "tampered"
    if risk_score <= low_thresh:          return "genuine"
    return "uncertain"


def _save(file_path, doc_type, result,
          high_thresh, low_thresh):
    risk   = result.get("risk",{}) or {}
    score  = risk.get("score",0) or 0
    level  = risk.get("level","LOW") or "LOW"
    flags  = result.get("flags",[])
    label  = _auto_label(score, flags, high_thresh, low_thresh)

    _exec("""
        INSERT INTO training_documents
        (file_path, file_name, doc_type, risk_score, risk_level,
         flags, full_result, label, label_source)
        VALUES (:fp,:fn,:dt,:rs,:rl,:fl,:fr,:lb,:ls)
        ON CONFLICT (file_path) DO UPDATE SET
            risk_score=EXCLUDED.risk_score,
            risk_level=EXCLUDED.risk_level,
            flags=EXCLUDED.flags,
            label=EXCLUDED.label
    """, {
        "fp": str(file_path),
        "fn": Path(file_path).name,
        "dt": doc_type,
        "rs": score,
        "rl": level,
        "fl": json.dumps(flags),
        "fr": json.dumps({k: v for k, v in result.items()
                          if k not in ("banking_analysis","full_result")}),
        "lb": label,
        "ls": "auto",
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_scan():
    """Analyse the document folder without processing anything."""
    print(f"\nSCANNING {DATA_DIR}\n{'─'*60}")

    type_counts = Counter()
    size_mb     = 0

    for f in DATA_DIR.iterdir():
        if not f.is_file(): continue
        dtype = _detect_type(f)
        type_counts[dtype] += 1
        try: size_mb += f.stat().st_size / 1_048_576
        except: pass

    print(f"{'Document Type':<25} {'Count':>8}  {'Recommended'}")
    print("─" * 60)

    recommendations = {
        "id_card":       "✓ TRAIN — core fraud detection",
        "bank_statement":"✓ TRAIN — banking intelligence",
        "cbc_report":    "✓ TRAIN — credit patterns",
        "payslip":       "✓ TRAIN — salary validation",
        "nid":           "✓ TRAIN — national ID PDFs",
        "image":         "~ REVIEW — may contain useful docs",
        "invoice":       "~ OPTIONAL — invoice patterns",
        "contract":      "─ SKIP — not a fraud target",
        "valuation":     "─ SKIP — not a fraud target",
        "loan_form":     "─ SKIP — internal forms",
        "pdf_other":     "~ REVIEW — check manually",
        "other":         "─ SKIP — unsupported format",
    }

    total = sum(type_counts.values())
    for dtype, count in sorted(type_counts.items(),
                                key=lambda x: -x[1]):
        rec = recommendations.get(dtype, "─ SKIP")
        print(f"  {dtype:<23} {count:>8}  {rec}")

    print(f"\n  {'TOTAL':<23} {total:>8}")
    print(f"  Total size: {size_mb:.1f} MB ({size_mb/1024:.1f} GB)")

    # Estimate processing time
    trainable = sum(type_counts.get(t,0) for t in
                    ["id_card","bank_statement","cbc_report","payslip","nid"])
    est_mins  = trainable * 2 / 60  # ~2 seconds per doc
    print(f"\nEstimated processing time for trainable docs ({trainable}):")
    print(f"  ~{est_mins:.0f} minutes on a single CPU core")
    print(f"\nRun:")
    print(f"  python train_targeted.py images  -- ID card images only ({type_counts.get('id_card',0)} docs, ~{type_counts.get('id_card',0)*2/60:.0f} min)")
    print(f"  python train_targeted.py pdfs    -- Banking PDFs only (~{est_mins:.0f} min)")
    print(f"  python train_targeted.py all     -- Everything trainable")


def cmd_images(batch_size: int = 100,
               high_thresh: int = 50,
               low_thresh:  int = 20):
    """Process all image files (ID cards)."""
    print(f"\nPROCESSING IMAGES\n{'─'*60}")
    _init_db()

    files     = [f for f in DATA_DIR.iterdir()
                 if f.is_file() and f.suffix in IMAGE_EXTS]
    ingested  = _already_ingested()
    new_files = [f for f in files if str(f) not in ingested]

    print(f"Total images    : {len(files)}")
    print(f"Already done    : {len(files)-len(new_files)}")
    print(f"To process      : {len(new_files)}")
    print(f"Thresholds      : tampered>={high_thresh}  genuine<={low_thresh}\n")

    if not new_files:
        print("All images already processed.")
        return

    prog = Progress(len(new_files), "images")

    for i, f in enumerate(new_files):
        dtype = _detect_type(f)
        try:
            result = _screen_image(str(f))
            _save(f, dtype, result, high_thresh, low_thresh)
            score  = result.get("risk",{}).get("score",0)
            level  = result.get("risk",{}).get("level","?")
            prog.update(True, f"{level}:{score} {f.name[:20]}")
        except Exception as e:
            _exec("""INSERT INTO training_documents (file_path,file_name,doc_type,notes)
                     VALUES (:fp,:fn,:dt,:n) ON CONFLICT DO NOTHING""",
                  {"fp":str(f),"fn":f.name,"dt":dtype,"n":f"error:{str(e)[:100]}"})
            prog.update(False, f"ERR {f.name[:20]}")

        # Checkpoint every batch
        if (i+1) % batch_size == 0:
            _print_interim_stats("id_card")

    prog.done()
    _print_interim_stats("id_card")


def cmd_pdfs(batch_size: int = 50,
             high_thresh: int = 50,
             low_thresh:  int = 20,
             types: list  = None):
    """Process banking PDFs."""
    target_types = types or [
        "bank_statement","cbc_report","payslip","nid","invoice"]

    print(f"\nPROCESSING PDFs\n{'─'*60}")
    print(f"Target types: {', '.join(target_types)}\n")
    _init_db()

    files    = [f for f in DATA_DIR.iterdir()
                if f.is_file() and f.suffix in PDF_EXTS]
    relevant = [(f, _detect_type(f)) for f in files]
    relevant = [(f,t) for f,t in relevant if t in target_types]
    ingested = _already_ingested()
    new_docs = [(f,t) for f,t in relevant if str(f) not in ingested]

    print(f"Total PDFs      : {len(files)}")
    print(f"Relevant type   : {len(relevant)}")
    print(f"Already done    : {len(relevant)-len(new_docs)}")
    print(f"To process      : {len(new_docs)}\n")

    if not new_docs:
        print("All relevant PDFs already processed.")
        return

    # Group by type for stats
    by_type = Counter(t for _,t in new_docs)
    for t,c in sorted(by_type.items()):
        print(f"  {t:<20} {c:>6} docs")
    print()

    prog = Progress(len(new_docs), "pdfs")

    for i, (f, dtype) in enumerate(new_docs):
        pdf_type = {
            "bank_statement": "bank_statement",
            "cbc_report":     "bank_statement",
            "payslip":        "payslip",
            "nid":            "bank_statement",
            "invoice":        "utility",
        }.get(dtype, "bank_statement")

        try:
            result = _screen_pdf(str(f), pdf_type)
            _save(f, dtype, result, high_thresh, low_thresh)
            score  = result.get("risk",{}).get("score",0)
            level  = result.get("risk",{}).get("level","?")
            prog.update(True, f"{level}:{score} {f.name[:20]}")
        except Exception as e:
            _exec("""INSERT INTO training_documents (file_path,file_name,doc_type,notes)
                     VALUES (:fp,:fn,:dt,:n) ON CONFLICT DO NOTHING""",
                  {"fp":str(f),"fn":f.name,"dt":dtype,
                   "n":f"error:{str(e)[:100]}"})
            prog.update(False, f"ERR {f.name[:20]}")

        if (i+1) % batch_size == 0:
            _print_interim_stats(dtype)

    prog.done()


def cmd_all(high_thresh: int = 50, low_thresh: int = 20):
    """Process all trainable document types."""
    cmd_images(high_thresh=high_thresh, low_thresh=low_thresh)
    cmd_pdfs(high_thresh=high_thresh, low_thresh=low_thresh)


def cmd_train(doc_type: str = "id_card", min_per_class: int = 5):
    """Train MobileNetV2 on processed documents."""
    import shutil

    print(f"\nTRAINING ML MODEL — {doc_type}\n{'─'*60}")

    rows = _q("""
        SELECT file_path, label FROM training_documents
        WHERE doc_type=:dt AND label IN ('genuine','tampered')
        ORDER BY label
    """, {"dt": doc_type})

    genuine  = [r for r in rows if r["label"]=="genuine"]
    tampered = [r for r in rows if r["label"]=="tampered"]

    print(f"Genuine  : {len(genuine)}")
    print(f"Tampered : {len(tampered)}")

    if len(genuine) < min_per_class or len(tampered) < min_per_class:
        print(f"\nNeed at least {min_per_class} per class. Run images/pdfs first.")
        return

    # Prepare ml_data dirs
    for d in ["genuine","tampered"]:
        p = ML_DIR / d
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    IMAGE_EXTS_SET = {".jpg",".jpeg",".png",".bmp",".jfif",
                      ".JPG",".PNG",".JPEG",".BMP"}
    copied = Counter()
    for row in genuine + tampered:
        src = Path(row["file_path"])
        if src.exists() and src.suffix in IMAGE_EXTS_SET:
            dst = ML_DIR / row["label"] / src.name
            shutil.copy2(src, dst)
            copied[row["label"]] += 1

    print(f"\nImages copied:")
    print(f"  genuine/  : {copied['genuine']}")
    print(f"  tampered/ : {copied['tampered']}")

    if copied["genuine"] < min_per_class:
        print(f"\nNot enough image files (need images, not PDFs, for ML training)")
        print(f"Process image files first: python train_targeted.py images")
        return

    print(f"\nStarting MobileNetV2 training (this takes 5-15 minutes)...\n")
    from ml_trainer import train_model, evaluate_model
    try:
        train_model()
        metrics = evaluate_model()
        print(f"\nResults:")
        print(f"  Accuracy : {metrics.get('val_accuracy',0):.1%}")
        print(f"  AUC      : {metrics.get('val_auc',0):.3f}")
        print(f"\nReload workers: docker compose restart worker-idcard")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback; traceback.print_exc()


def cmd_anomaly():
    """Retrain anomaly detector."""
    print(f"\nTRAINING ANOMALY DETECTOR\n{'─'*60}")
    from anomaly_detect import train_anomaly_detector
    result = train_anomaly_detector(min_samples=30)
    if result.get("status") == "trained":
        print(f"Trained on {result['training_samples']} samples")
        print(f"Threshold: {result['score_threshold']:.4f}")
    else:
        print(f"Result: {result}")


def cmd_report():
    """Show training report."""
    print(f"\nTRAINING REPORT\n{'─'*60}\n")

    # Overall counts
    rows = _q("""
        SELECT doc_type, label, COUNT(*) AS cnt,
               ROUND(AVG(risk_score)::numeric,1) AS avg_score
        FROM training_documents
        GROUP BY doc_type, label ORDER BY doc_type, label
    """)

    by_type = defaultdict(dict)
    for r in rows:
        by_type[r["doc_type"]][r["label"] or "unlabeled"] = r["cnt"]

    print(f"{'Type':<22} {'Genuine':>8} {'Tampered':>9} {'Uncertain':>10} {'Unlabeled':>10} {'Total':>6}")
    print("─" * 70)

    grand = Counter()
    for dtype, labels in sorted(by_type.items()):
        g  = labels.get("genuine",0)
        t  = labels.get("tampered",0)
        u  = labels.get("uncertain",0)
        ul = labels.get("unlabeled",0)
        tot= g+t+u+ul
        print(f"  {dtype:<20} {g:>8} {t:>9} {u:>10} {ul:>10} {tot:>6}")
        grand["genuine"]   += g
        grand["tampered"]  += t
        grand["uncertain"] += u
        grand["total"]     += tot

    print("─" * 70)
    print(f"  {'TOTAL':<20} {grand['genuine']:>8} {grand['tampered']:>9} "
          f"{grand['uncertain']:>10} {'':>10} {grand['total']:>6}")

    # Top flags
    print(f"\nTop 15 flags in TAMPERED documents:")
    print("─" * 55)
    flag_rows = _q("""
        SELECT flag, COUNT(*) AS cnt FROM (
            SELECT jsonb_array_elements_text(flags) AS flag
            FROM training_documents WHERE label='tampered'
            AND flags IS NOT NULL
        ) sub GROUP BY flag ORDER BY cnt DESC LIMIT 15
    """)
    if flag_rows:
        max_cnt = flag_rows[0]["cnt"]
        for r in flag_rows:
            bar = "█" * int(r["cnt"]/max_cnt*25)
            print(f"  {r['flag']:<42} {bar:<25} {r['cnt']:>5}")

    # Risk score distribution
    print(f"\nRisk score distribution:")
    dist = _q("""
        SELECT CASE
            WHEN risk_score=0    THEN '0 (no flags)'
            WHEN risk_score<20   THEN '1-19 (genuine)'
            WHEN risk_score<50   THEN '20-49 (uncertain)'
            WHEN risk_score<100  THEN '50-99 (tampered)'
            ELSE '100+ (high risk)' END AS bucket,
            COUNT(*) AS cnt
        FROM training_documents
        WHERE risk_score IS NOT NULL
        GROUP BY bucket ORDER BY MIN(risk_score)
    """)
    for r in dist:
        bar = "█" * min(int(r["cnt"]/(grand["total"] or 1)*50),50)
        print(f"  {r['bucket']:<25} {bar:<50} {r['cnt']:>6}")

    # Model status
    print(f"\nModel Status:")
    try:
        from ml_trainer import load_model
        model, meta = load_model()
        if model:
            print(f"  ML Classifier   : TRAINED — accuracy={meta.get('val_accuracy',0):.1%} AUC={meta.get('val_auc',0):.3f}")
        else:
            print(f"  ML Classifier   : not trained — run: python train_targeted.py train")
    except:
        print(f"  ML Classifier   : not available")

    try:
        from anomaly_detect import get_anomaly_stats
        s = get_anomaly_stats()
        if s.get("status") != "no_model":
            print(f"  Anomaly Detector: TRAINED — {s.get('training_samples',0)} samples")
        else:
            print(f"  Anomaly Detector: not trained — run: python train_targeted.py anomaly")
    except:
        print(f"  Anomaly Detector: not available")

    print(f"\nNext steps:")
    if grand["uncertain"] > 0:
        print(f"  Review {grand['uncertain']} uncertain docs: python train_pipeline.py review")
    print(f"  Train ML model   : python train_targeted.py train")
    print(f"  Train anomaly    : python train_targeted.py anomaly")
    print(f"  Reload workers   : docker compose restart worker-idcard")


def _print_interim_stats(doc_type: str):
    rows = _q("""
        SELECT label, COUNT(*) AS cnt FROM training_documents
        WHERE doc_type=:dt AND label IS NOT NULL
        GROUP BY label
    """, {"dt": doc_type})
    stats = {r["label"]: r["cnt"] for r in rows}
    print(f"\n  [{doc_type}] genuine={stats.get('genuine',0)} "
          f"tampered={stats.get('tampered',0)} "
          f"uncertain={stats.get('uncertain',0)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Targeted Training Pipeline")
    p.add_argument("cmd", choices=["scan","images","pdfs","all","train","anomaly","report"])
    p.add_argument("--high",     type=int, default=50,       help="Tampered threshold (default 50)")
    p.add_argument("--low",      type=int, default=20,       help="Genuine threshold (default 20)")
    p.add_argument("--doc-type", default="id_card",          help="Doc type for training")
    p.add_argument("--min",      type=int, default=5,        help="Min samples per class")
    p.add_argument("--batch",    type=int, default=50,       help="Checkpoint interval")
    args = p.parse_args()

    print(f"KYC Training Pipeline — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Data folder: {DATA_DIR}\n")

    if   args.cmd == "scan":    cmd_scan()
    elif args.cmd == "images":  cmd_images(args.batch, args.high, args.low)
    elif args.cmd == "pdfs":    cmd_pdfs(args.batch, args.high, args.low)
    elif args.cmd == "all":     cmd_all(args.high, args.low)
    elif args.cmd == "train":   cmd_train(args.doc_type, args.min)
    elif args.cmd == "anomaly": cmd_anomaly()
    elif args.cmd == "report":  cmd_report()

if __name__ == "__main__":
    main()
