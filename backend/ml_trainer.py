"""
ml_trainer.py — ML model training pipeline for Cambodia ID document classification
Uses heavy augmentation to work with small datasets (as few as 3 images per class).

Workflow:
  1. python ml_trainer.py setup
  2. Copy images to ml_data/genuine/ and ml_data/tampered/
  3. python ml_trainer.py train
  4. python ml_trainer.py evaluate
"""
import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

log = logging.getLogger("ml_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

DATA_DIR   = Path("ml_data")
MODEL_DIR  = Path("ml_models")
MODEL_PATH = MODEL_DIR / "khm_id_classifier.h5"
META_PATH  = MODEL_DIR / "khm_id_classifier_meta.json"

IMG_SIZE       = (224, 224)
BATCH_SIZE     = 8       # small batch for small dataset
EPOCHS         = 30
AUGMENT_FACTOR = 50      # generate 50 augmented versions per original image
CLASSES        = ["genuine", "tampered"]


def _check_deps():
    try:
        import tensorflow as tf
        import cv2
        return True
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  HEAVY AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment_image(img: np.ndarray) -> list:
    """
    Generate AUGMENT_FACTOR variations of a single image.
    Covers: rotation, flip, brightness, contrast, blur, noise,
            crop, perspective warp, colour jitter, JPEG compression artefacts.
    Returns list of augmented images (numpy arrays, float32, 0-1).
    """
    import cv2

    results = []
    h, w = img.shape[:2]

    for _ in range(AUGMENT_FACTOR):
        aug = img.copy()

        # ── Geometric ────────────────────────────────────────────────────────
        # Random rotation ±8 degrees
        angle = random.uniform(-8, 8)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Random horizontal flip (50%)
        if random.random() > 0.5:
            aug = cv2.flip(aug, 1)

        # Random crop and resize (zoom 85-100%)
        scale  = random.uniform(0.85, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0     = random.randint(0, h - ch)
        x0     = random.randint(0, w - cw)
        aug    = aug[y0:y0+ch, x0:x0+cw]
        aug    = cv2.resize(aug, (w, h))

        # Perspective warp (slight tilt — simulates camera angle)
        if random.random() > 0.5:
            pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            jitter = 15
            pts2 = np.float32([
                [random.randint(0, jitter),      random.randint(0, jitter)],
                [w - random.randint(0, jitter),  random.randint(0, jitter)],
                [random.randint(0, jitter),      h - random.randint(0, jitter)],
                [w - random.randint(0, jitter),  h - random.randint(0, jitter)],
            ])
            M2  = cv2.getPerspectiveTransform(pts1, pts2)
            aug = cv2.warpPerspective(aug, M2, (w, h), borderMode=cv2.BORDER_REFLECT)

        # ── Colour / lighting ─────────────────────────────────────────────────
        aug = aug.astype(np.float32)

        # Brightness jitter ±25%
        brightness = random.uniform(0.75, 1.25)
        aug = np.clip(aug * brightness, 0, 255)

        # Contrast jitter
        contrast = random.uniform(0.8, 1.2)
        mean     = aug.mean()
        aug      = np.clip((aug - mean) * contrast + mean, 0, 255)

        # Channel-wise colour jitter (simulate different scanner/camera)
        for c in range(3):
            aug[:,:,c] = np.clip(aug[:,:,c] * random.uniform(0.9, 1.1), 0, 255)

        # Random gamma
        gamma = random.uniform(0.7, 1.3)
        aug   = np.clip(np.power(aug / 255.0, gamma) * 255, 0, 255)

        aug = aug.astype(np.uint8)

        # ── Noise / blur ──────────────────────────────────────────────────────
        choice = random.random()
        if choice < 0.3:
            # Gaussian blur (simulates out-of-focus scan)
            ksize = random.choice([3, 5])
            aug   = cv2.GaussianBlur(aug, (ksize, ksize), 0)
        elif choice < 0.5:
            # Gaussian noise (simulates scanner noise)
            noise = np.random.normal(0, random.uniform(3, 12), aug.shape).astype(np.int16)
            aug   = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif choice < 0.65:
            # Sharpen (simulates over-sharpened scan)
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            aug    = cv2.filter2D(aug, -1, kernel)

        # JPEG compression artefacts (simulates photo of photo / re-saving)
        if random.random() > 0.6:
            quality = random.randint(50, 90)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buf = cv2.imencode(".jpg", aug, encode_param)
            aug    = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # ── Final resize and normalise ────────────────────────────────────────
        aug = cv2.resize(aug, IMG_SIZE)
        aug = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
        aug = aug.astype(np.float32) / 255.0

        results.append(aug)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataset():
    import cv2

    X, y   = [], []
    counts = {}

    for label, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            cls_dir.mkdir(parents=True)
            log.warning(f"{cls_dir} created but empty")
            counts[cls] = 0
            continue

        images = (list(cls_dir.glob("*.jpg")) +
                  list(cls_dir.glob("*.jpeg")) +
                  list(cls_dir.glob("*.png")))

        log.info(f"  {cls}: {len(images)} original image(s) found")
        counts[cls] = len(images)

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning(f"  Cannot read: {img_path.name}")
                continue

            img = cv2.resize(img, IMG_SIZE)

            # add original
            orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            X.append(orig)
            y.append(label)

            # add augmented versions
            log.info(f"  Augmenting {img_path.name} → {AUGMENT_FACTOR} variations...")
            for aug in augment_image(img):
                X.append(aug)
                y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), counts


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        input_shape = (*IMG_SIZE, 3),
        include_top = False,
        weights     = "imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x      = base(inputs, training=False)
    x      = layers.GlobalAveragePooling2D()(x)
    x      = layers.Dropout(0.5)(x)           # higher dropout for small dataset
    x      = layers.Dense(64, activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
    x      = layers.Dropout(0.4)(x)
    out    = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, out)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    if not _check_deps():
        return

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("KHM ID Card Fraud Classifier — Training")
    log.info(f"Augmentation factor: {AUGMENT_FACTOR}x per image")
    log.info("=" * 60)

    log.info("Loading and augmenting images...")
    X, y, counts = prepare_dataset()

    if len(X) == 0:
        log.error("No images found. Run 'setup' first and add images.")
        return

    total = len(X)
    log.info(f"Total samples after augmentation: {total}")
    log.info(f"  Genuine:  {np.sum(y==0)} samples from {counts.get('genuine',0)} originals")
    log.info(f"  Tampered: {np.sum(y==1)} samples from {counts.get('tampered',0)} originals")

    # warn about small dataset
    if counts.get("genuine", 0) < 10 or counts.get("tampered", 0) < 10:
        log.warning("Small dataset detected — model accuracy may be limited.")
        log.warning("Collect more images when possible for better results.")

    # split — use small val set for tiny datasets
    test_size = 0.15 if total < 100 else 0.20
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    cw     = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = {i: w for i, w in enumerate(cw)}
    log.info(f"Class weights: {cw_dict}")

    # build model
    log.info("Building MobileNetV2 model...")
    model, base = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8,
            restore_best_weights=True, mode="max", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1
        ),
    ]

    # ── Phase 1: train head only ──────────────────────────────────────────────
    log.info("Phase 1: Training classification head (base frozen)...")
    history1 = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = 15,
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks,
        class_weight    = cw_dict,
        verbose         = 1,
    )

    # ── Phase 2: fine-tune top layers ─────────────────────────────────────────
    log.info("Phase 2: Fine-tuning top 30 layers of MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer = tf.keras.optimizers.Adam(5e-6),  # very low LR for fine-tune
        loss      = "binary_crossentropy",
        metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=10,
            restore_best_weights=True, mode="max", verbose=1
        ),
    ]

    history2 = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks2,
        class_weight    = cw_dict,
        verbose         = 1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    log.info("=" * 60)
    log.info(f"Final Results:")
    log.info(f"  val_accuracy : {val_acc:.3f}")
    log.info(f"  val_auc      : {val_auc:.3f}")
    log.info(f"  val_loss     : {val_loss:.4f}")

    if val_auc < 0.65:
        log.warning("AUC < 0.65 — model is weak. Collect more images for better results.")
    elif val_auc < 0.80:
        log.info("AUC 0.65-0.80 — model is usable but limited. More data will help.")
    else:
        log.info("AUC > 0.80 — good model performance!")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    log.info(f"Model saved → {MODEL_PATH}")

    meta = {
        "trained_at":         datetime.utcnow().isoformat(),
        "original_images":    counts,
        "total_after_augment": int(total),
        "augment_factor":     AUGMENT_FACTOR,
        "classes":            CLASSES,
        "image_size":         list(IMG_SIZE),
        "val_accuracy":       round(float(val_acc), 4),
        "val_auc":            round(float(val_auc), 4),
        "val_loss":           round(float(val_loss), 4),
        "warning":            "Small dataset — collect 50+ images per class for production use" if counts.get("genuine", 0) < 50 else "",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    log.info(f"Metadata saved → {META_PATH}")
    log.info("Training complete!")


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    if not MODEL_PATH.exists():
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
        meta  = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        log.info(f"ML model loaded — auc: {meta.get('val_auc','?')} "
                 f"acc: {meta.get('val_accuracy','?')}")
        return model, meta
    except Exception as e:
        log.warning(f"Could not load ML model: {e}")
        return None, None


def predict(image_path: str, model, meta: dict) -> dict:
    import cv2

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"ml_error": "image_read_failed"}

        size = tuple(meta.get("image_size", list(IMG_SIZE)))
        img  = cv2.resize(img, size)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = img.astype(np.float32) / 255.0
        img  = np.expand_dims(img, 0)

        score = float(model.predict(img, verbose=0)[0][0])

        return {
            "ml_tamper_score":  round(score, 4),
            "ml_prediction":    "tampered" if score > 0.5 else "genuine",
            "ml_confidence":    round(abs(score - 0.5) * 2, 4),
            "ml_model_auc":     meta.get("val_auc"),
            "ml_model_warning": meta.get("warning", ""),
            "ml_trained_at":    meta.get("trained_at", ""),
        }
    except Exception as e:
        return {"ml_error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def setup():
    for cls in CLASSES:
        d = DATA_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}/")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created: {MODEL_DIR}/")
    print()
    print("Next steps:")
    print(f"  1. Copy genuine KHM ID images  → {DATA_DIR}/genuine/")
    print(f"  2. Copy tampered/fake ID images → {DATA_DIR}/tampered/")
    print(f"  3. Run: python ml_trainer.py train")
    print()
    print(f"Note: Even 3 images per class will work with {AUGMENT_FACTOR}x augmentation,")
    print("      but 50+ per class gives much better accuracy.")


def evaluate():
    if not META_PATH.exists():
        print("No trained model found. Run: python ml_trainer.py train")
        return
    meta = json.loads(META_PATH.read_text())
    print()
    print("=" * 50)
    print("ML Model Report")
    print("=" * 50)
    print(f"  Trained at     : {meta.get('trained_at','?')}")
    print(f"  Original images: {meta.get('original_images',{})}")
    print(f"  Total samples  : {meta.get('total_after_augment','?')} (after {meta.get('augment_factor','?')}x augmentation)")
    print(f"  Val Accuracy   : {meta.get('val_accuracy','?')}")
    print(f"  Val AUC        : {meta.get('val_auc','?')}")
    print(f"  Val Loss       : {meta.get('val_loss','?')}")
    if meta.get("warning"):
        print(f"  ⚠ Warning      : {meta['warning']}")
    print("=" * 50)


def test_image(image_path: str):
    """Quick test: run inference on a single image."""
    model, meta = load_model()
    if model is None:
        print("No model found. Train first.")
        return
    result = predict(image_path, model, meta)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KHM ID Card ML Trainer")
    parser.add_argument("command",
                        choices=["setup", "train", "evaluate", "test"],
                        help="setup | train | evaluate | test <image_path>")
    parser.add_argument("image_path", nargs="?",
                        help="Image path for test command")
    args = parser.parse_args()

    if args.command == "setup":    setup()
    elif args.command == "train":  train()
    elif args.command == "evaluate": evaluate()
    elif args.command == "test":
        if not args.image_path:
            print("Usage: python ml_trainer.py test <image_path>")
        else:
            test_image(args.image_path)
