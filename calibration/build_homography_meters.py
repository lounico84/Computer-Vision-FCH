import numpy as np
import cv2
import os
from config import Settings
s = Settings()
analytics_cfg = s.analytics

# Projekt-Root und Pfade
H_NPY    = str(s.paths.homography_npy)
PITCH_IMG = str(s.paths.pitch_image)
OUT_NPZ   = str(s.paths.homography_npz)

# Echte Feldgröße in Metern
FIELD_LENGTH_M = analytics_cfg.pitch_length
FIELD_WIDTH_M  = analytics_cfg.pitch_width

# 1) H in Pitch-Pixel laden
H_px = np.load(H_NPY)

# 2) Pitch-Bild laden, um Pixelgröße zu bekommen
img = cv2.imread(PITCH_IMG)
if img is None:
    raise FileNotFoundError(f"Pitch-Bild nicht gefunden: {PITCH_IMG}")
h, w = img.shape[:2]
print("Pitch-Bildgröße:", w, "x", h)

# 3) Skalierung: Pitch-Pixel -> Meter
S = np.array([
    [FIELD_LENGTH_M / w, 0.0,                 0.0],
    [0.0,                FIELD_WIDTH_M / h,   0.0],
    [0.0,                0.0,                 1.0],
], dtype=np.float32)

# 4) Gesamthomographie: Kamera-Pixel -> Meter
H_m = S @ H_px
H_inv = np.linalg.inv(H_m)

np.savez(OUT_NPZ, H=H_m, H_inv=H_inv)
print("Neue Homographie in Metern gespeichert unter:", OUT_NPZ)
print("H_m =\n", H_m)