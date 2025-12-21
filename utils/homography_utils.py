import os
import numpy as np
import cv2

from config import Settings

s = Settings()

# Resolve and store the configured homography (.npz) path for later lazy loading
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CALIBRATION_FILE = os.path.join(s.paths.homography_npz)

_H = None
_H_inv = None
_loaded = False


# Lazily load the homography matrices (H and H_inv) from disk on first access
def _load_homography():
    global _H, _H_inv, _loaded
    if _loaded:
        return

    _loaded = True
    if not os.path.exists(CALIBRATION_FILE):
        print(f"[homography_utils] No calibration file found at {CALIBRATION_FILE}")
        return

    data = np.load(CALIBRATION_FILE)
    _H = data["H"]
    _H_inv = data.get("H_inv", None)
    print(f"[homography_utils] Homography loaded from {CALIBRATION_FILE}")


# Check whether a valid homography matrix could be loaded successfully
def is_homography_available() -> bool:
    _load_homography()
    return _H is not None


# Convert camera image pixel coordinates into pitch coordinates in meters using H
def pixel_to_pitch(x: float, y: float):
    _load_homography()
    if _H is None:
        return np.nan, np.nan

    pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, _H)
    X = float(dst[0, 0, 0])
    Y = float(dst[0, 0, 1])
    return X, Y


# Convert pitch coordinates in meters back into camera image pixel coordinates using H_inv
def pitch_to_pixel(X: float, Y: float):
    _load_homography()
    if _H_inv is None:
        return np.nan, np.nan

    pts = np.array([[[float(X), float(Y)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, _H_inv)
    x = float(dst[0, 0, 0])
    y = float(dst[0, 0, 1])
    return x, y