import os
import numpy as np
import cv2

from config import Settings

s = Settings()

# Pfad relativ zum Projekt-Root ermitteln
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CALIBRATION_FILE = os.path.join(s.paths.homography_npz)

_H = None
_H_inv = None
_loaded = False


def _load_homography():
    """
    Lädt H und H_inv aus der NPZ-Datei (lazy, nur beim ersten Zugriff).
    """
    global _H, _H_inv, _loaded
    if _loaded:
        return

    _loaded = True
    if not os.path.exists(CALIBRATION_FILE):
        print(f"[homography_utils] Keine Kalibrierdatei gefunden unter {CALIBRATION_FILE}")
        return

    data = np.load(CALIBRATION_FILE)
    _H = data["H"]
    _H_inv = data.get("H_inv", None)
    print(f"[homography_utils] Homographie geladen aus {CALIBRATION_FILE}")


def is_homography_available() -> bool:
    """
    True, wenn eine gültige Homographie geladen werden konnte.
    """
    _load_homography()
    return _H is not None


def pixel_to_pitch(x: float, y: float):
    """
    Transformiert einen Bildpunkt (Pixel) in Spielfeld-Koordinaten (Meter).

    Rückgabe: (X, Y) in Metern
    Falls keine Homographie vorhanden ist: (np.nan, np.nan)
    """
    _load_homography()
    if _H is None:
        return np.nan, np.nan

    pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, _H)
    X = float(dst[0, 0, 0])
    Y = float(dst[0, 0, 1])
    return X, Y


def pitch_to_pixel(X: float, Y: float):
    """
    Transformiert einen Spielfeldpunkt (Meter) in Bildkoordinaten (Pixel).

    Rückgabe: (x, y) in Pixel
    Falls keine Homographie vorhanden ist: (np.nan, np.nan)
    """
    _load_homography()
    if _H_inv is None:
        return np.nan, np.nan

    pts = np.array([[[float(X), float(Y)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, _H_inv)
    x = float(dst[0, 0, 0])
    y = float(dst[0, 0, 1])
    return x, y