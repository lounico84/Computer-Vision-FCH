# field_mapping.py
import numpy as np
import cv2
from config import Settings
s = Settings()
analytics_cfg = s.analytics

PITCH_IMG_FILE = str(s.paths.pitch_image)
H_FILE         = str(s.paths.homography_npy)

# echte Feldgröße in Metern
FIELD_LENGTH_M = analytics_cfg.pitch_length
FIELD_WIDTH_M  = analytics_cfg.pitch_width

class FieldMapper:
    def __init__(self):
        self.H = np.load(H_FILE)
        self.pitch_img = cv2.imread(PITCH_IMG_FILE)
        if self.pitch_img is None:
            raise FileNotFoundError("pitch_map.png nicht gefunden")
        self.h, self.w = self.pitch_img.shape[:2]

    def cam_to_pitch_px(self, x, y):
        """Kamera-Pixel -> Pitch-Pixel (im pitch_map.png)."""
        p = np.array([x, y, 1.0], dtype=np.float32)
        p_ = self.H @ p
        X = p_[0] / p_[2]
        Y = p_[1] / p_[2]
        return float(X), float(Y)

    def pitch_px_to_m(self, X, Y):
        """Pitch-Pixel -> Meterkoordinaten (0..100m, 0..60m)."""
        mx = X / self.w * FIELD_LENGTH_M
        my = Y / self.h * FIELD_WIDTH_M
        return mx, my

    def cam_to_meters(self, x, y):
        """Direkt Kamera-Pixel -> Meter auf dem Feld."""
        X, Y = self.cam_to_pitch_px(x, y)
        return self.pitch_px_to_m(X, Y)