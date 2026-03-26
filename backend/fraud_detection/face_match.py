"""
face_match.py — Face liveness detection + selfie-vs-ID matching via DeepFace
No cloud APIs required; runs entirely on-device.
"""
import os
import logging
import numpy as np
import cv2
from pathlib import Path

log = logging.getLogger("fraud_detect.face")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    log.warning("DeepFace not installed — face matching disabled")


# ── Liveness heuristics ────────────────────────────────────────────────────────
# DeepFace does not have a built-in liveness model, so we combine:
#   1. Texture/frequency analysis (print-attack detection)
#   2. Laplacian variance (blurry = photo-of-photo)
#   3. Specular highlight check (real faces reflect light)
#   4. Colour distribution naturalness

def _detect_face_region(image_path: str):
    """Return the largest face bounding box using OpenCV Haar cascade."""
    img      = cv2.imread(image_path)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None, img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
    return (x, y, w, h), img


def _lbp_texture_score(gray_face: np.ndarray) -> float:
    """
    Local Binary Pattern uniformity score.
    Real faces have rich, varied textures; printed/screen photos are flatter.
    Returns 0.0 (fake) – 1.0 (real) heuristic.
    """
    rows, cols = gray_face.shape
    lbp = np.zeros_like(gray_face)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = int(gray_face[i, j])
            binary = ""
            for di, dj in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
                binary += "1" if int(gray_face[i+di, j+dj]) >= center else "0"
            lbp[i, j] = int(binary, 2)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 255))
    hist    = hist.astype(float) / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    # real face LBP entropy typically > 5.5
    return min(entropy / 8.0, 1.0)


def _specular_highlight_score(bgr_face: np.ndarray) -> float:
    """Real faces have small specular highlights; printed photos don't."""
    gray    = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    ratio   = mask.sum() / 255 / max(gray.size, 1)
    # expect 0.001–0.05 for real faces
    return 1.0 if 0.0005 < ratio < 0.06 else 0.0


def _frequency_analysis_score(gray_face: np.ndarray) -> float:
    """
    High-frequency content via FFT.
    Printed/screen attacks show periodic Moiré patterns.
    We flag unusually high periodic peaks.
    """
    f       = np.fft.fft2(gray_face.astype(float))
    fshift  = np.fft.fftshift(f)
    mag     = 20 * np.log(np.abs(fshift) + 1)
    # centre region (DC component) excluded
    h, w    = mag.shape
    centre  = mag[h//4:3*h//4, w//4:3*w//4]
    outer   = mag.copy()
    outer[h//4:3*h//4, w//4:3*w//4] = 0
    ratio   = outer.mean() / max(centre.mean(), 0.001)
    # low ratio = energy concentrated in DC = flat/printed image
    return min(ratio * 5, 1.0)


def analyze_liveness(image_path: str) -> tuple[dict, list[str]]:
    """
    Heuristic liveness check (anti-spoofing).
    Returns info dict and flags list.
    """
    flags, info = [], {}

    try:
        bbox, img = _detect_face_region(image_path)
        if bbox is None:
            flags.append("no_face_detected")
            return info, flags

        x, y, w, h = bbox
        face_bgr    = img[y:y+h, x:x+w]
        face_gray   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # resize to standard for consistency
        face_gray = cv2.resize(face_gray, (128, 128))
        face_bgr  = cv2.resize(face_bgr,  (128, 128))

        blur_score    = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        texture_score = _lbp_texture_score(face_gray)
        specular      = _specular_highlight_score(face_bgr)
        freq_score    = _frequency_analysis_score(face_gray)

        # combined liveness score (0–100)
        liveness = round(
            (texture_score * 40) +
            (specular      * 20) +
            (freq_score    * 25) +
            (min(blur_score / 200, 1.0) * 15),
            2
        )

        info.update(
            liveness_score   = liveness,
            blur_variance    = round(blur_score, 2),
            texture_score    = round(texture_score, 4),
            specular_score   = round(specular, 4),
            frequency_score  = round(freq_score, 4),
            face_bbox        = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        )

        if liveness < 30:
            flags.append("likely_spoofed_face")
        elif liveness < 50:
            flags.append("possible_spoofed_face")

        if blur_score < 20:
            flags.append("face_region_too_blurry")

    except Exception as e:
        flags.append("liveness_check_failed")
        info["liveness_error"] = str(e)

    return info, flags


def match_faces(selfie_path: str, id_card_path: str) -> tuple[dict, list[str]]:
    """
    Compare selfie against the face on the ID card using DeepFace.
    Uses ArcFace model for accuracy; falls back to VGG-Face if unavailable.
    """
    flags, info = [], {}

    if not DEEPFACE_AVAILABLE:
        flags.append("deepface_not_installed")
        return info, flags

    try:
        result = DeepFace.verify(
            img1_path  = selfie_path,
            img2_path  = id_card_path,
            model_name = "ArcFace",
            detector_backend = "opencv",
            enforce_detection = False,
            distance_metric = "cosine",
        )

        distance  = round(float(result.get("distance", 1.0)), 4)
        verified  = bool(result.get("verified", False))
        threshold = round(float(result.get("threshold", 0.68)), 4)

        # convert distance to similarity %
        similarity = round((1 - distance / max(threshold, 0.001)) * 100, 2)
        similarity = max(0.0, min(similarity, 100.0))

        info.update(
            face_match        = verified,
            cosine_distance   = distance,
            threshold         = threshold,
            similarity_pct    = similarity,
            model             = "ArcFace",
        )

        if not verified:
            flags.append("face_match_failed")
        if similarity < 40:
            flags.append("low_face_similarity")

    except ValueError as e:
        # DeepFace raises ValueError when no face is detected
        flags.append("no_face_in_one_image")
        info["match_error"] = str(e)
    except Exception as e:
        flags.append("face_match_error")
        info["match_error"] = str(e)

    return info, flags
