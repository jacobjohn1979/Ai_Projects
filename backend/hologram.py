"""
hologram.py — Security feature & hologram detection for ID cards
Techniques:
  1. HSV iridescence scan  — holograms shift colour under different angles;
                              genuine IDs show rainbow HSV variance
  2. Frequency-domain ghost pattern — holograms leave micro-print FFT signatures
  3. Micro-text density check — genuine IDs have high-frequency micro-text regions
  4. Background pattern regularity — security guilloché patterns are periodic
  5. Retro-reflective zone brightness — holographic patches are brighter
"""
import logging
import numpy as np
import cv2

log = logging.getLogger("fraud_detect.hologram")


# ── 1. HSV iridescence ─────────────────────────────────────────────────────────

def _hsv_iridescence(img_bgr: np.ndarray) -> dict:
    """
    Holograms cause localised hue variance.
    We measure hue standard deviation in small blocks and flag if unusually uniform
    (no hologram) or has a distinct iridescent patch.
    """
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue      = hsv[:, :, 0].astype(float)
    sat      = hsv[:, :, 1].astype(float)

    # high-saturation mask — holograms are saturated
    sat_mask = sat > 80
    irid_hue = hue[sat_mask] if sat_mask.any() else hue.ravel()

    hue_std  = float(np.std(irid_hue))
    hue_mean = float(np.mean(irid_hue))

    return {
        "hue_std":         round(hue_std, 2),
        "hue_mean":        round(hue_mean, 2),
        "high_sat_pixels": int(sat_mask.sum()),
    }


# ── 2. FFT micro-print signature ───────────────────────────────────────────────

def _fft_periodicity(gray: np.ndarray) -> dict:
    """
    Genuine ID security backgrounds (guilloché, micro-print) produce
    periodic FFT peaks. A blank/forged background is flat.
    """
    f      = np.fft.fft2(gray.astype(float))
    fshift = np.fft.fftshift(f)
    mag    = np.abs(fshift)

    h, w   = mag.shape
    # exclude DC component (centre)
    mag[h//2-5:h//2+5, w//2-5:w//2+5] = 0

    top_peaks   = np.sort(mag.ravel())[-50:]
    peak_energy = float(top_peaks.mean())
    total_energy = float(mag.mean())
    peak_ratio  = peak_energy / max(total_energy, 1.0)

    return {
        "fft_peak_ratio":  round(peak_ratio, 4),
        "fft_peak_energy": round(peak_energy, 2),
    }


# ── 3. Micro-text density ──────────────────────────────────────────────────────

def _micro_text_density(gray: np.ndarray) -> dict:
    """
    Genuine IDs contain micro-text visible only at high resolution.
    We use a high-pass filter and count fine detail pixels.
    """
    kernel     = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    high_pass  = cv2.filter2D(gray, -1, kernel)
    _, thresh  = cv2.threshold(high_pass, 30, 255, cv2.THRESH_BINARY)
    density    = float(thresh.sum() / 255) / max(gray.size, 1)

    return {"micro_text_density": round(density, 4)}


# ── 4. Retro-reflective patch brightness ──────────────────────────────────────

def _bright_patch_check(img_bgr: np.ndarray) -> dict:
    """
    Holographic patches on genuine IDs are brighter than surrounding areas.
    We look for localised brightness anomalies.
    """
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w      = gray.shape
    block_h   = h // 4
    block_w   = w // 4

    block_means = []
    for r in range(4):
        for c in range(4):
            block = gray[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
            block_means.append(float(block.mean()))

    global_mean = float(np.mean(block_means))
    max_block   = float(np.max(block_means))
    brightness_ratio = max_block / max(global_mean, 1.0)

    return {
        "brightness_ratio":    round(brightness_ratio, 4),
        "max_block_brightness": round(max_block, 2),
        "global_brightness":   round(global_mean, 2),
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def analyze_hologram(image_path: str) -> tuple[dict, list[str]]:
    """
    Full hologram / security feature analysis.
    Returns (info_dict, flags_list).
    """
    flags, info = [], {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return info, ["image_read_failed"]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # run all checks
        hsv_info    = _hsv_iridescence(img)
        fft_info    = _fft_periodicity(gray)
        micro_info  = _micro_text_density(gray)
        bright_info = _bright_patch_check(img)

        info.update(**hsv_info, **fft_info, **micro_info, **bright_info)

        # ── scoring rules ────────────────────────────────────────────────────

        # Low hue variance → no iridescence → hologram likely absent/removed
        if hsv_info["hue_std"] < 15 and hsv_info["high_sat_pixels"] < 500:
            flags.append("no_iridescence_detected")

        # Very low saturated pixels → printed on plain paper
        if hsv_info["high_sat_pixels"] < 200:
            flags.append("low_colour_saturation")

        # Flat FFT → no security background pattern
        if fft_info["fft_peak_ratio"] < 5:
            flags.append("missing_security_background_pattern")

        # Very low micro-text density → likely blank/forged background
        if micro_info["micro_text_density"] < 0.002:
            flags.append("low_micro_text_density")

        # Unusually high brightness in one block → possible hologram sticker present
        if bright_info["brightness_ratio"] > 1.8:
            info["holographic_patch_detected"] = True
        else:
            info["holographic_patch_detected"] = False
            flags.append("holographic_patch_not_detected")

        # Combined strong signal: multiple missing features
        strong_missing = sum([
            "no_iridescence_detected" in flags,
            "missing_security_background_pattern" in flags,
            "low_micro_text_density" in flags,
        ])
        if strong_missing >= 2:
            flags.append("multiple_security_features_absent")

    except Exception as e:
        flags.append("hologram_analysis_failed")
        info["hologram_error"] = str(e)

    return info, flags
