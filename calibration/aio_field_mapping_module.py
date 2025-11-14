# field_mapping.py
import numpy as np
import cv2

PITCH_IMG_FILE = "project/Computer-Vision-FCH/calibration/fch_fussballfeld.jpg"
H_FILE         = "project/Computer-Vision-FCH/calibration/aio_homography_cam_to_map.npy"

# echte Feldgröße in Metern
FIELD_LENGTH_M = 100.0
FIELD_WIDTH_M  = 60.0

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
        """Pitch-Pixel -> Meterkoordinaten (0..105m, 0..68m)."""
        mx = X / self.w * FIELD_LENGTH_M
        my = Y / self.h * FIELD_WIDTH_M
        return mx, my

    def cam_to_meters(self, x, y):
        """Direkt Kamera-Pixel -> Meter auf dem Feld."""
        X, Y = self.cam_to_pitch_px(x, y)
        return self.pitch_px_to_m(X, Y)