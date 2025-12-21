import numpy as np
import cv2
import os
from config import Settings

# Load settings to resolve calibration paths and real pitch dimensions from a single source of truth
s = Settings()
analytics_cfg = s.analytics

# Resolve homography and pitch assets from config to keep the export reproducible
H_NPY    = str(s.paths.homography_npy)
PITCH_IMG = str(s.paths.pitch_image)
OUT_NPZ   = str(s.paths.homography_npz)

# Cache real-world pitch dimensions (meters) for pixel-to-metric scaling
FIELD_LENGTH_M = analytics_cfg.pitch_length
FIELD_WIDTH_M  = analytics_cfg.pitch_width

# Load pixel-space homography (camera pixels -> pitch-image pixels) as the conversion base
H_px = np.load(H_NPY)

# Load pitch image to derive pixel-to-meter scaling factors from its resolution
img = cv2.imread(PITCH_IMG)
if img is None:
    raise FileNotFoundError(f"Pitch-Bild nicht gefunden: {PITCH_IMG}")
h, w = img.shape[:2]
print("Pitch-Bildgröße:", w, "x", h)

# Build a scale matrix mapping pitch-image pixels into metric pitch coordinates
S = np.array([
    [FIELD_LENGTH_M / w, 0.0,                 0.0],
    [0.0,                FIELD_WIDTH_M / h,   0.0],
    [0.0,                0.0,                 1.0],
], dtype=np.float32)

# Compose the final metric homography (camera pixels -> meters) for downstream analytics
H_m = S @ H_px
H_inv = np.linalg.inv(H_m)

# Persist both forward and inverse transforms for pipeline use and back-projection diagnostics
np.savez(OUT_NPZ, H=H_m, H_inv=H_inv)
print("Neue Homographie in Metern gespeichert unter:", OUT_NPZ)
print("H_m =\n", H_m)