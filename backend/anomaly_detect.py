import os, json, logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
load_dotenv()
log = logging.getLogger("fraud_detect.anomaly")

DATABASE_URL  = os.getenv("DATABASE_URL","postgresql://postgres:password@postgres:5432/fraud_detect")
engine        = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal  = sessionmaker(bind=engine)
MODEL_DIR     = Path("ml_models")
MODEL_PATH    = MODEL_DIR / "anomaly_detector.pkl"
META_PATH     = MODEL_DIR / "anomaly_detector_meta.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _extract_features(result):
    ela   = result.get("ela",{}) or {}
    noise = result.get("noise_analysis",{}) or {}
    reg   = result.get("region_info",{}) or {}
    holo  = result.get("hologram",{}) or {}
    copy  = result.get("copy_move",{}) or {}
    tmpl  = result.get("template_match",{}) or {}
    risk  = result.get("risk",{}) or {}
    flags = result.get("flags",[])
    return [
        float(ela.get("ela_mean_diff",0) or 0),
        float(ela.get("ela_std_diff",0) or 0),
        float(ela.get("ela_max_diff",0) or 0),
        float(noise.get("noise_variance_ratio",1) or 1),
        float(np.mean(noise.get("noise_quadrant_variances",[0])) if noise.get("noise_quadrant_variances") else 0),
        float(reg.get("photo_sharpness",0) or 0),
        float(reg.get("text_sharpness",0) or 0),
        float(reg.get("mrz_sharpness",0) or 0),
        float(reg.get("color_temperature_delta",0) or 0),
        float(holo.get("hue_std",0) or 0),
        float(holo.get("fft_peak_ratio",0) or 0),
        float(holo.get("micro_text_density",0) or 0),
        float(1 if holo.get("holographic_patch_detected") else 0),
        float(copy.get("sift_suspicious_matches",0) or 0),
        float(copy.get("total_keypoints",0) or 0),
        float(tmpl.get("keyword_ratio",0) or 0),
        float(tmpl.get("aspect_ratio_delta_pct",0) or 0),
        float(1 if tmpl.get("khmer_detected") else 0),
        float(risk.get("score",0) or 0),
        float(len(flags)),
    ]

def train_anomaly_detector(min_samples=30):
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        import pickle
    except ImportError:
        return {"error":"scikit-learn not installed"}

    # First try training_documents table (our labeled set)
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT full_result FROM training_documents
            WHERE label='genuine' AND full_result IS NOT NULL
            ORDER BY id DESC LIMIT 500
        """)).fetchall()
        features = []
        for row in rows:
            result = row.full_result
            if isinstance(result, str):
                try: result = json.loads(result)
                except: continue
            if not result: continue
            try:
                feat = _extract_features(result)
                if any(f != 0 for f in feat): features.append(feat)
            except: pass

        # Fall back to screening_logs if not enough
        if len(features) < min_samples:
            rows2 = db.execute(text("""
                SELECT full_result FROM screening_logs
                WHERE risk_level='LOW' AND doc_type='id_card'
                AND full_result IS NOT NULL
                ORDER BY screened_at DESC LIMIT 500
            """)).fetchall()
            for row in rows2:
                result = row.full_result
                if isinstance(result, str):
                    try: result = json.loads(result)
                    except: continue
                if not result: continue
                try:
                    feat = _extract_features(result)
                    if any(f != 0 for f in feat): features.append(feat)
                except: pass
    finally:
        db.close()

    if len(features) < min_samples:
        return {"error":"insufficient_data","found":len(features),"needed":min_samples}

    import pickle
    X = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    model  = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_s)
    scores = model.score_samples(X_s)
    with open(MODEL_PATH,"wb") as f:
        pickle.dump({"model":model,"scaler":scaler}, f)
    meta = {
        "trained_at":       datetime.utcnow().isoformat(),
        "training_samples": len(X),
        "feature_count":    X.shape[1],
        "score_mean":       float(np.mean(scores)),
        "score_std":        float(np.std(scores)),
        "score_threshold":  float(np.percentile(scores,5)),
        "contamination":    0.05,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    return {"status":"trained", **meta}

def load_anomaly_detector():
    if not MODEL_PATH.exists(): return None, None
    try:
        import pickle
        with open(MODEL_PATH,"rb") as f: model_dict = pickle.load(f)
        meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        return model_dict, meta
    except Exception as e:
        log.warning(f"Could not load anomaly detector: {e}")
        return None, None

_cached_model = None
_cached_meta  = None

def _get_model():
    global _cached_model, _cached_meta
    if _cached_model is None: _cached_model, _cached_meta = load_anomaly_detector()
    return _cached_model, _cached_meta

def detect_anomaly(result):
    flags, info = [], {}
    model_dict, meta = _get_model()
    if model_dict is None:
        info["anomaly_status"] = "no_model_trained"
        return info, []
    try:
        X = np.nan_to_num(np.array([_extract_features(result)], dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
        X_s        = model_dict["scaler"].transform(X)
        score      = float(model_dict["model"].score_samples(X_s)[0])
        prediction = model_dict["model"].predict(X_s)[0]
        threshold  = meta.get("score_threshold",-0.5)
        is_anomaly = prediction == -1
        severity   = "high" if score < threshold*1.5 else ("medium" if is_anomaly else "low")
        info["anomaly_score"]     = round(score,4)
        info["anomaly_threshold"] = round(threshold,4)
        info["is_anomaly"]        = is_anomaly
        info["anomaly_severity"]  = severity
        info["model_trained_at"]  = meta.get("trained_at","unknown")
        info["training_samples"]  = meta.get("training_samples",0)
        if is_anomaly:
            flags.append(f"anomaly_detected:{severity}")
            if severity == "high": flags.append("anomaly_high_severity")
    except Exception as e:
        log.error(f"Anomaly detection failed: {e}")
        info["anomaly_error"] = str(e)
    return info, flags

def get_anomaly_stats():
    _, meta = load_anomaly_detector()
    return meta if meta else {"status":"no_model"}
