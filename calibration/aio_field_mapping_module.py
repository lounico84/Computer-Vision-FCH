import numpy as np
import cv2
from config import Settings

# Load project settings to centralize paths and pitch dimensions
s = Settings()
analytics_cfg = s.analytics

# Resolve asset paths from configuration for consistent deployment across environments
PITCH_IMG_FILE = str(s.paths.pitch_image)
H_FILE         = str(s.paths.homography_npy)

# Cache real-world pitch dimensions (meters) for pixel-to-metric scaling
FIELD_LENGTH_M = analytics_cfg.pitch_length
FIELD_WIDTH_M  = analytics_cfg.pitch_width


class FieldMapper:
    # Provide camera-to-pitch mapping utilities backed by homography and pitch-image scaling
    def __init__(self):
        # Load homography and pitch image once to avoid repeated I/O during frame processing
        self.H = np.load(H_FILE)
        self.pitch_img = cv2.imread(PITCH_IMG_FILE)
        if self.pitch_img is None:
            raise FileNotFoundError("pitch_map.png nicht gefunden")
        self.h, self.w = self.pitch_img.shape[:2]
        # Keep pitch image bounds for robust out-of-range rejection after projection

    def cam_to_pitch_px(self, x, y):
        # Apply homography to map camera pixel coordinates into pitch-image coordinates
        p = np.array([x, y, 1.0], dtype=np.float32)
        p_ = self.H @ p
        X = p_[0] / p_[2]
        Y = p_[1] / p_[2]
        return float(X), float(Y)

    def pitch_px_to_m(self, X, Y):
        # Reject projected points outside the pitch image to prevent invalid metric outputs
        if not (0.0 <= X <= self.w and 0.0 <= Y <= self.h):
            return np.nan, np.nan

        # Convert pitch-image pixel coordinates into meters using linear scaling factors
        mx = X / self.w * FIELD_LENGTH_M
        my = Y / self.h * FIELD_WIDTH_M

        # Enforce real-pitch bounds as a final safety filter against edge artifacts
        if not (0.0 <= mx <= FIELD_LENGTH_M and 0.0 <= my <= FIELD_WIDTH_M):
            return np.nan, np.nan
        return mx, my

    def cam_to_meters(self, x, y):
        # Provide a single-call transformation from camera pixels to real-world pitch coordinates
        X, Y = self.cam_to_pitch_px(x, y)
        return self.pitch_px_to_m(X, Y)