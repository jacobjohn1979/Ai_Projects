"""
train_pipeline.py — Complete Document Training Pipeline

Stages:
  1. ingest   — scan folder, screen all documents through KYC engine
  2. label    — auto-label using risk scores + clustering
  3. train    — train MobileNetV2 classifier per document type
  4. anomaly  — retrain Isolation Forest on genuine documents
  5. report   — generate training report with quality metrics
  6. review   — interactive review of uncertain labels

Usage:
  python train_pipeline.py ingest  --folder /path/to/documents
  python train_pipeline.py label   --threshold 50
  python train_pipeline.py train   --doc-type id_card
  python train_pipeline.py anomaly
  python train_pipeline.py report
  python train_pipeline.py review
  python train_pipeline.py all     --folder /path/to/documents
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("train_pipeline")

DATABASE_URL = os.getenv("DATABASE_URL",
    "postgresql://postgres:password@postgres:5432/fraud_detect")
engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

ML_DATA_DIR   = Path("ml_data")
TRAIN_LOG_DIR = Path("training_logs")
ML_DATA_DIR.mkdir(exist_ok=True)
TRAIN_LOG_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
PDF_EXTS   = {".pdf"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _q(sql, params={}):
    db = SessionLocal()
    try:
        return [dict(r._mapping) for r in db.execute(text(sql), params)]
    except Exception as e:
        log.error(f"Query: {e}"); return []
    finally:
        db.close()


def _exec(sql, params={}):
    db = SessionLocal()
    try:
        db.execute(text(sql), params); db.commit()
    except Exception as e:
        db.rollback(); log.error(f"Exec: {e}")
    finally:
        db.close()


def _sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def _init_training_table():
    _exec("""
        CREATE TABLE IF NOT EXISTS training_documents (
            id           SERIAL PRIMARY KEY,
            file_path    TEXT UNIQUE,
            file_name    VARCHAR(255),
            file_hash    VARCHAR(64),
            doc_type     VARCHAR(50),
            risk_score   INTEGER,
            risk_level   VARCHAR(10),
            flags        JSON,
            full_result  JSON,
            label        VARCHAR(20),
            label_source VARCHAR(30),
            cluster_id   INTEGER,
            ingested_at  TIMESTAMP DEFAULT NOW(),
            labeled_at   TIMESTAMP,
            notes        TEXT
        )
    """)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: INGEST
# ═══════════════════════════════════════════════════════════════════════════════

def stage_ingest(folder: str, limit: int = 0):
    """
    Scan folder recursively, screen every document through the KYC engine,
    store results in training_documents table.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: INGEST — scanning {folder}")
    print(f"{'='*60}")

    _init_training_table()

    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    # Collect all files
    all_files = []
    for ext in IMAGE_EXTS | PDF_EXTS:
        all_files.extend(folder_path.rglob(f"*{ext}"))
        all_files.extend(folder_path.rglob(f"*{ext.upper()}"))

    all_files = list(set(all_files))
    if limit: all_files = all_files[:limit]

    print(f"Found {len(all_files)} documents")

    # Check which are already ingested
    existing = set(r["file_hash"] for r in _q(
        "SELECT file_hash FROM training_documents WHERE file_hash IS NOT NULL"))
    print(f"Already ingested: {len(existing)}")

    new_files = [f for f in all_files if _sha256(str(f)) not in existing]
    print(f"New to process:  {len(new_files)}\n")

    stats   = Counter()
    results = []

    for i, file_path in enumerate(new_files, 1):
        ext      = file_path.suffix.lower()
        sha      = _sha256(str(file_path))
        doc_type = _detect_doc_type(file_path)

        print(f"[{i:4d}/{len(new_files)}] {file_path.name[:45]:<45} "
              f"type={doc_type:<12}", end=" ", flush=True)

        try:
            result = _screen_document(file_path, doc_type)
            risk_score = (result.get("risk",{}) or {}).get("score",0) or \
                         result.get("risk_score",0) or 0
            risk_level = (result.get("risk",{}) or {}).get("level","LOW") or \
                         result.get("risk_level","LOW") or "LOW"
            flags      = result.get("flags",[])

            print(f"risk={risk_level:<8} score={risk_score:3d} flags={len(flags)}")

            _exec("""
                INSERT INTO training_documents
                (file_path, file_name, file_hash, doc_type, risk_score,
                 risk_level, flags, full_result)
                VALUES (:fp, :fn, :fh, :dt, :rs, :rl, :fl, :fr)
                ON CONFLICT (file_path) DO UPDATE SET
                    risk_score=EXCLUDED.risk_score,
                    risk_level=EXCLUDED.risk_level,
                    flags=EXCLUDED.flags,
                    full_result=EXCLUDED.full_result
            """, {
                "fp": str(file_path), "fn": file_path.name, "fh": sha,
                "dt": doc_type, "rs": risk_score, "rl": risk_level,
                "fl": json.dumps(flags), "fr": json.dumps(result),
            })

            stats[risk_level] += 1
            stats["total"]    += 1

        except Exception as e:
            print(f"ERROR: {e}")
            stats["error"] += 1

    print(f"\n{'─'*60}")
    print(f"Ingestion complete:")
    print(f"  Total processed : {stats['total']}")
    print(f"  HIGH risk       : {stats.get('HIGH',0)}")
    print(f"  MEDIUM risk     : {stats.get('MEDIUM',0)}")
    print(f"  LOW risk        : {stats.get('LOW',0)}")
    print(f"  Errors          : {stats.get('error',0)}")


def _detect_doc_type(file_path: Path) -> str:
    """Detect document type from filename and extension."""
    name = file_path.name.lower()
    ext  = file_path.suffix.lower()

    if ext == ".pdf":
        if any(k in name for k in ["statement","bank","account","cbc","acleda"]): return "bank_statement"
        if any(k in name for k in ["payslip","salary","wage","pay"]):              return "payslip"
        if any(k in name for k in ["tax","revenue","irs"]):                        return "tax"
        if any(k in name for k in ["utility","electric","water","bill"]):          return "utility"
        return "pdf"

    # Images
    if any(k in name for k in ["id","national","khm","identity","card"]): return "id_card"
    if any(k in name for k in ["passport","travel"]):                      return "passport"
    if any(k in name for k in ["selfie","face","photo","portrait"]):       return "selfie"
    return "image"


def _screen_document(file_path: Path, doc_type: str) -> dict:
    """Screen a document through the KYC engine."""
    ext = file_path.suffix.lower()

    if doc_type in ("id_card","passport") and ext in IMAGE_EXTS:
        from screening import (
            compute_ela, analyze_noise, analyze_regions,
            detect_copy_move, analyze_hologram,
            match_template, extract_id_fields,
            score_id_card, check_velocity
        )
        import cv2, numpy as np

        img = cv2.imread(str(file_path))
        if img is None:
            return {"error": "could_not_read_image", "flags": [], "risk": {"level":"LOW","score":0}}

        # Rotate if portrait
        h, w = img.shape[:2]
        if h < w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        ela_info   = compute_ela(img)
        noise_info = analyze_noise(img)
        region_info= analyze_regions(img)
        cm_info    = detect_copy_move(img)
        holo_info  = analyze_hologram(img)
        tmpl_info  = match_template(img, str(file_path))

        try:    field_info = extract_id_fields(img)
        except: field_info = {}

        velocity    = check_velocity(None, "train_" + hashlib.md5(
                        str(file_path).encode()).hexdigest()[:8])
        all_flags   = (ela_info.get("flags",[]) + noise_info.get("flags",[]) +
                      region_info.get("flags",[]) + cm_info.get("flags",[]) +
                      holo_info.get("flags",[]) + tmpl_info.get("flags",[]) +
                      velocity["flags"])

        risk_score, risk_level = score_id_card(all_flags)
        return {
            "flags": all_flags, "field_info": field_info,
            "ela": ela_info, "noise_analysis": noise_info,
            "region_info": region_info, "copy_move": cm_info,
            "hologram": holo_info, "template_match": tmpl_info,
            "risk": {"level": risk_level, "score": risk_score},
        }

    elif ext == ".pdf":
        from pdf_banking import analyze_banking_pdf
        info, flags, score, level = analyze_banking_pdf(
            str(file_path), "", 0, doc_type)
        return {"flags": flags, "banking_analysis": info,
                "risk": {"level": level, "score": score}}

    else:
        from screening import compute_ela, analyze_noise, score_image
        import cv2
        img = cv2.imread(str(file_path))
        if img is None:
            return {"flags":[], "risk":{"level":"LOW","score":0}}
        ela_info   = compute_ela(img)
        noise_info = analyze_noise(img)
        all_flags  = ela_info.get("flags",[]) + noise_info.get("flags",[])
        score, level = score_image(all_flags)
        return {"flags": all_flags, "ela": ela_info,
                "risk": {"level": level, "score": score}}


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: AUTO-LABEL
# ═══════════════════════════════════════════════════════════════════════════════

def stage_label(high_threshold: int = 50, low_threshold: int = 20):
    """
    Auto-label documents as genuine/tampered based on risk scores and flags.

    Rules:
      risk_score >= high_threshold  → tampered (suspicious)
      risk_score <= low_threshold   → genuine
      between                       → uncertain (needs manual review)
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2: AUTO-LABEL")
    print(f"  tampered threshold  : score >= {high_threshold}")
    print(f"  genuine threshold   : score <= {low_threshold}")
    print(f"{'='*60}\n")

    rows = _q("""
        SELECT id, file_path, doc_type, risk_score, risk_level, flags
        FROM training_documents WHERE label IS NULL ORDER BY id
    """)

    print(f"Documents to label: {len(rows)}")
    stats = Counter()

    for row in rows:
        score  = row.get("risk_score") or 0
        flags  = row.get("flags") or []
        if isinstance(flags, str):
            try: flags = json.loads(flags)
            except: flags = []

        # Determine label
        if score >= high_threshold:
            label = "tampered"
        elif score <= low_threshold:
            label = "genuine"
        else:
            label = "uncertain"

        # Override: certain HIGH-confidence flags always mean tampered
        definite_tampered = {
            "mrz_checksum_failed", "face_mismatch", "balance_math_inconsistency",
            "template_shared_across_applicants", "fraud_ring_suspected",
            "benfords_law_violation", "future_dates_detected",
        }
        if any(f in flags for f in definite_tampered):
            label = "tampered"

        _exec("""
            UPDATE training_documents
            SET label=:l, label_source='auto', labeled_at=NOW()
            WHERE id=:id
        """, {"l": label, "id": row["id"]})
        stats[label] += 1

    total = sum(stats.values())
    print(f"\nLabeling complete:")
    print(f"  Genuine    : {stats['genuine']:4d}  ({stats['genuine']/max(total,1)*100:.0f}%)")
    print(f"  Tampered   : {stats['tampered']:4d}  ({stats['tampered']/max(total,1)*100:.0f}%)")
    print(f"  Uncertain  : {stats['uncertain']:4d}  ({stats['uncertain']/max(total,1)*100:.0f}%)")
    print(f"\nNote: Review 'uncertain' documents manually with:")
    print(f"  python train_pipeline.py review")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: TRAIN ML CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def stage_train(doc_type: str = "id_card", min_samples: int = 10):
    """
    Train MobileNetV2 classifier for a specific document type.
    Uses auto-labeled documents from training_documents table.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 3: TRAIN ML CLASSIFIER — {doc_type}")
    print(f"{'='*60}\n")

    # Get labeled image documents
    rows = _q("""
        SELECT file_path, label FROM training_documents
        WHERE doc_type=:dt AND label IN ('genuine','tampered')
        AND label_source IS NOT NULL
        ORDER BY label, id
    """, {"dt": doc_type})

    genuine  = [r for r in rows if r["label"] == "genuine"]
    tampered = [r for r in rows if r["label"] == "tampered"]

    print(f"Genuine  samples: {len(genuine)}")
    print(f"Tampered samples: {len(tampered)}")

    if len(genuine) < min_samples:
        print(f"\nNot enough genuine samples (need {min_samples}, have {len(genuine)})")
        print("Try lowering --min-samples or ingesting more documents")
        return False

    if len(tampered) < min_samples:
        print(f"\nNot enough tampered samples (need {min_samples}, have {len(tampered)})")
        print("The system will use augmentation but accuracy may be limited")

    # Copy files to ml_data directory
    genuine_dir  = ML_DATA_DIR / "genuine"
    tampered_dir = ML_DATA_DIR / "tampered"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing
    for f in genuine_dir.iterdir():  f.unlink()
    for f in tampered_dir.iterdir(): f.unlink()

    copied = {"genuine": 0, "tampered": 0}
    for row in genuine:
        src = Path(row["file_path"])
        if src.exists() and src.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(src, genuine_dir / src.name)
            copied["genuine"] += 1

    for row in tampered:
        src = Path(row["file_path"])
        if src.exists() and src.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(src, tampered_dir / src.name)
            copied["tampered"] += 1

    print(f"\nCopied to ml_data/:")
    print(f"  genuine/  : {copied['genuine']} images")
    print(f"  tampered/ : {copied['tampered']} images")

    if copied["genuine"] < min_samples:
        print(f"\nERROR: Not enough image files found on disk")
        return False

    # Run ml_trainer
    print(f"\nStarting MobileNetV2 training...")
    print(f"This may take 5-15 minutes on CPU...\n")

    from ml_trainer import train_model, evaluate_model, MODEL_PATH

    try:
        history = train_model()
        print(f"\nTraining complete!")

        # Evaluate
        metrics = evaluate_model()
        print(f"\nEvaluation results:")
        print(f"  Accuracy  : {metrics.get('accuracy',0):.1%}")
        print(f"  AUC       : {metrics.get('auc',0):.3f}")
        print(f"  Precision : {metrics.get('precision',0):.3f}")
        print(f"  Recall    : {metrics.get('recall',0):.3f}")

        # Log to DB
        _exec("""
            INSERT INTO training_documents (file_path, file_name, doc_type, notes)
            VALUES (:fp, 'TRAINING_RUN', :dt, :notes)
            ON CONFLICT DO NOTHING
        """, {
            "fp": f"training_run_{datetime.utcnow().isoformat()}",
            "dt": doc_type,
            "notes": json.dumps({
                "trained_at": datetime.utcnow().isoformat(),
                "genuine_count": copied["genuine"],
                "tampered_count": copied["tampered"],
                "metrics": metrics,
            }),
        })

        return True

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: RETRAIN ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def stage_anomaly():
    """Retrain Isolation Forest using genuine documents from training set."""
    print(f"\n{'='*60}")
    print(f"STAGE 4: RETRAIN ANOMALY DETECTOR")
    print(f"{'='*60}\n")

    # Count genuine results in screening_logs
    count = _q("""
        SELECT COUNT(*) AS cnt FROM screening_logs
        WHERE risk_level='LOW' AND doc_type='id_card'
    """)
    n = count[0]["cnt"] if count else 0
    print(f"Genuine screenings in DB: {n}")

    if n < 30:
        print(f"Need at least 30. Currently have {n}.")
        print("Run ingest first so genuine documents get screened and saved.")
        return False

    from anomaly_detect import train_anomaly_detector
    print("Training Isolation Forest...")
    result = train_anomaly_detector(min_samples=30)
    if result.get("status") == "trained":
        print(f"Anomaly detector trained on {result['training_samples']} samples")
        print(f"Score threshold: {result['score_threshold']:.4f}")
        return True
    else:
        print(f"Training failed: {result}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 5: REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def stage_report():
    """Generate a training quality report."""
    print(f"\n{'='*60}")
    print(f"STAGE 5: TRAINING REPORT")
    print(f"{'='*60}\n")

    # Overall stats
    rows = _q("""
        SELECT doc_type, label, COUNT(*) AS cnt,
               ROUND(AVG(risk_score)::numeric,1) AS avg_score
        FROM training_documents
        WHERE label IS NOT NULL
        GROUP BY doc_type, label ORDER BY doc_type, label
    """)

    by_type = defaultdict(dict)
    for r in rows:
        by_type[r["doc_type"]][r["label"]] = {
            "count": r["cnt"], "avg_score": r["avg_score"]}

    print(f"{'Doc Type':<20} {'Genuine':>8} {'Tampered':>9} {'Uncertain':>10} {'Total':>6}")
    print("─" * 60)

    for dtype, labels in sorted(by_type.items()):
        g = labels.get("genuine",   {}).get("count",0)
        t = labels.get("tampered",  {}).get("count",0)
        u = labels.get("uncertain", {}).get("count",0)
        print(f"{dtype:<20} {g:>8} {t:>9} {u:>10} {g+t+u:>6}")

    # Top flags in tampered docs
    print(f"\nTop 15 fraud flags in TAMPERED documents:")
    print("─" * 50)

    flag_rows = _q("""
        SELECT flag, COUNT(*) AS cnt FROM (
            SELECT jsonb_array_elements_text(flags::jsonb) AS flag
            FROM training_documents WHERE label='tampered'
            AND flags IS NOT NULL AND flags != 'null'
        ) sub GROUP BY flag ORDER BY cnt DESC LIMIT 15
    """)

    for r in flag_rows:
        bar = "█" * min(int(r["cnt"] / max(flag_rows[0]["cnt"],1) * 20), 20)
        print(f"  {r['flag']:<45} {bar} {r['cnt']}")

    # Uncertain documents needing review
    uncertain_count = _q("""
        SELECT COUNT(*) AS cnt FROM training_documents WHERE label='uncertain'
    """)
    uc = uncertain_count[0]["cnt"] if uncertain_count else 0
    print(f"\nUncertain documents needing manual review: {uc}")
    if uc > 0:
        print(f"Run: python train_pipeline.py review")

    # ML model status
    from ml_trainer import load_model, MODEL_PATH
    model, meta = load_model()
    print(f"\nML Model status:")
    if model:
        print(f"  Status    : Trained")
        print(f"  Accuracy  : {meta.get('val_accuracy',0):.1%}")
        print(f"  AUC       : {meta.get('val_auc',0):.3f}")
        print(f"  Trained   : {meta.get('trained_at','unknown')[:16]}")
    else:
        print(f"  Status    : Not trained yet")
        print(f"  Run       : python train_pipeline.py train --doc-type id_card")

    # Anomaly detector status
    from anomaly_detect import get_anomaly_stats
    astats = get_anomaly_stats()
    print(f"\nAnomaly Detector status:")
    if astats.get("status") != "no_model":
        print(f"  Status    : Trained")
        print(f"  Samples   : {astats.get('training_samples',0)}")
        print(f"  Trained   : {astats.get('trained_at','unknown')[:16]}")
    else:
        print(f"  Status    : Not trained yet")

    # Save report
    report_path = TRAIN_LOG_DIR / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "document_stats": {dt: dict(lbls) for dt, lbls in by_type.items()},
        "top_flags": [dict(r) for r in flag_rows],
        "uncertain_count": uc,
        "ml_model": meta if model else {"status": "not_trained"},
        "anomaly_detector": astats,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 6: MANUAL REVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def stage_review(limit: int = 20):
    """Interactive review of uncertain documents."""
    print(f"\n{'='*60}")
    print(f"STAGE 6: MANUAL REVIEW — uncertain documents")
    print(f"{'='*60}\n")

    rows = _q("""
        SELECT id, file_name, doc_type, risk_score, flags
        FROM training_documents WHERE label='uncertain'
        ORDER BY risk_score DESC LIMIT :lim
    """, {"lim": limit})

    if not rows:
        print("No uncertain documents to review.")
        return

    print(f"Reviewing {len(rows)} uncertain documents")
    print(f"Commands: g=genuine  t=tampered  s=skip  q=quit\n")

    reviewed = 0
    for row in rows:
        flags = row.get("flags") or []
        if isinstance(flags, str):
            try: flags = json.loads(flags)
            except: flags = []

        print(f"\n{'─'*50}")
        print(f"File  : {row['file_name']}")
        print(f"Type  : {row['doc_type']}")
        print(f"Score : {row['risk_score']}")
        print(f"Flags : {', '.join(flags[:5])}{' (+more)' if len(flags)>5 else ''}")

        choice = input("Label [g/t/s/q]: ").strip().lower()

        if choice == "q":
            break
        elif choice == "g":
            _exec("""UPDATE training_documents
                     SET label='genuine', label_source='manual', labeled_at=NOW()
                     WHERE id=:id""", {"id": row["id"]})
            print("Labeled: genuine")
            reviewed += 1
        elif choice == "t":
            _exec("""UPDATE training_documents
                     SET label='tampered', label_source='manual', labeled_at=NOW()
                     WHERE id=:id""", {"id": row["id"]})
            print("Labeled: tampered")
            reviewed += 1
        else:
            print("Skipped")

    print(f"\nManual review complete — {reviewed} documents labeled")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Document Training Pipeline")
    parser.add_argument("stage", choices=["ingest","label","train","anomaly","report","review","all"])
    parser.add_argument("--folder",       default="",         help="Document folder for ingest")
    parser.add_argument("--limit",        type=int, default=0, help="Max docs to ingest")
    parser.add_argument("--threshold",    type=int, default=50, help="Risk score threshold for tampered label")
    parser.add_argument("--low-threshold",type=int, default=20, help="Risk score threshold for genuine label")
    parser.add_argument("--doc-type",     default="id_card",  help="Document type to train")
    parser.add_argument("--min-samples",  type=int, default=10, help="Minimum samples per class")
    args = parser.parse_args()

    print(f"\nKYC FRAUD DETECTION — TRAINING PIPELINE")
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    if args.stage == "ingest" or args.stage == "all":
        if not args.folder:
            print("ERROR: --folder required for ingest stage")
            sys.exit(1)
        stage_ingest(args.folder, args.limit)

    if args.stage == "label" or args.stage == "all":
        stage_label(args.threshold, args.low_threshold)

    if args.stage == "train" or args.stage == "all":
        stage_train(args.doc_type, args.min_samples)

    if args.stage == "anomaly" or args.stage == "all":
        stage_anomaly()

    if args.stage == "report" or args.stage == "all":
        stage_report()

    if args.stage == "review":
        stage_review()

    print(f"\nDone: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    main()
