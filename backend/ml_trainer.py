"""
ml_trainer.py — ML model training pipeline for Cambodia ID document classification
Uses a lightweight CNN (MobileNetV2) fine-tuned on your real KHM ID images.

Workflow:
  1. Place images in:
       ml_data/genuine/   ← real authentic KHM IDs
       ml_data/tampered/  ← confirmed tampered/fake IDs
  2. Run: python ml_trainer.py train
  3. Model saved to: ml_models/khm_id_classifier.h5
  4. The screening pipeline loads this model automatically if it exists.

Minimum recommended: 50 images per class (100 total)
Ideal: 200+ images per class
"""
import os
import sys
import json
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

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS     = 20
CLASSES    = ["genuine", "tampered"]


# ── Check dependencies ────────────────────────────────────────────────────────

def _check_deps():
    try:
        import tensorflow as tf
        import cv2
        return True
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        log.error("Install: pip install tensorflow opencv-python")
        return False


# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_dataset():
    """Load and preprocess images from ml_data/genuine/ and ml_data/tampered/"""
    import cv2

    X, y = [], []
    counts = {}

    for label, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            log.warning(f"Directory not found: {cls_dir} — creating it")
            cls_dir.mkdir(parents=True)
            continue

        images = list(cls_dir.glob("*.jpg")) + \
                 list(cls_dir.glob("*.jpeg")) + \
                 list(cls_dir.glob("*.png"))

        log.info(f"  {cls}: {len(images)} images found")
        counts[cls] = len(images)

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning(f"  Could not read: {img_path}")
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label)

    if not X:
        log.error("No images found. Add images to ml_data/genuine/ and ml_data/tampered/")
        return None, None, counts

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)

    return X, y, counts


def augment_data(X, y):
    """Simple augmentation — flip, brightness, contrast."""
    import tensorflow as tf
    X_aug, y_aug = list(X), list(y)

    for img, label in zip(X, y):
        # horizontal flip
        X_aug.append(np.fliplr(img))
        y_aug.append(label)
        # slight brightness
        bright = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
        X_aug.append(bright.astype(np.float32))
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """MobileNetV2 fine-tuned for binary classification (genuine vs tampered)."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        input_shape = (*IMG_SIZE, 3),
        include_top = False,
        weights     = "imagenet",
    )

    # freeze base layers initially
    base.trainable = False

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # 0=genuine, 1=tampered

    model = Model(inputs, outputs)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    if not _check_deps():
        return

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading images...")
    X, y, counts = prepare_dataset()

    if X is None:
        return

    total = len(X)
    log.info(f"Total images: {total} ({counts})")

    if total < 20:
        log.warning("Very few images — model quality will be low. Aim for 100+.")

    # augment
    log.info("Augmenting data...")
    X, y = augment_data(X, y)
    log.info(f"After augmentation: {len(X)} samples")

    # split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # build model
    log.info("Building MobileNetV2 model...")
    model, base = build_model()

    # callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=5,
            restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    # phase 1: train head only
    log.info("Phase 1: Training classification head...")
    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = min(EPOCHS // 2, 10),
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # phase 2: fine-tune top layers of base
    log.info("Phase 2: Fine-tuning top layers...")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-5),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # evaluate
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    log.info(f"Final — val_accuracy: {val_acc:.3f}  val_auc: {val_auc:.3f}")

    # save model
    model.save(str(MODEL_PATH))
    log.info(f"Model saved to {MODEL_PATH}")

    # save metadata
    meta = {
        "trained_at":   datetime.utcnow().isoformat(),
        "total_images": total,
        "classes":      CLASSES,
        "image_size":   IMG_SIZE,
        "val_accuracy": round(float(val_acc), 4),
        "val_auc":      round(float(val_auc), 4),
        "counts":       counts,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved to {META_PATH}")
    log.info("Training complete!")


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model():
    """Load trained model if it exists. Returns (model, meta) or (None, None)."""
    if not MODEL_PATH.exists():
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
        meta  = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        log.info(f"ML model loaded — accuracy: {meta.get('val_accuracy', '?')}")
        return model, meta
    except Exception as e:
        log.warning(f"Could not load ML model: {e}")
        return None, None


def predict(image_path: str, model, meta: dict) -> dict:
    """
    Run ML inference on a document image.
    Returns dict with score and prediction.
    """
    import cv2

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"ml_error": "image_read_failed"}

        img = cv2.resize(img, tuple(meta.get("image_size", IMG_SIZE)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        score = float(model.predict(img, verbose=0)[0][0])

        return {
            "ml_tamper_score":  round(score, 4),       # 0=genuine, 1=tampered
            "ml_prediction":    "tampered" if score > 0.5 else "genuine",
            "ml_confidence":    round(abs(score - 0.5) * 2, 4),  # 0-1
            "ml_model_version": meta.get("trained_at", "unknown"),
            "ml_val_accuracy":  meta.get("val_accuracy"),
        }
    except Exception as e:
        return {"ml_error": str(e)}


# ── Setup helper ───────────────────────────────────────────────────────────────

def setup():
    """Create folder structure for training data."""
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


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate():
    """Show model metadata and performance."""
    if not META_PATH.exists():
        print("No trained model found. Run: python ml_trainer.py train")
        return
    meta = json.loads(META_PATH.read_text())
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KHM ID Card ML Trainer")
    parser.add_argument("command", choices=["setup", "train", "evaluate"],
                        help="setup: create folders | train: train model | evaluate: show metrics")
    args = parser.parse_args()

    if args.command == "setup":
        setup()
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
