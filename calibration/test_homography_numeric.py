# calibration/test_homography_numeric.py

import os
import sys
from pathlib import Path

import numpy as np

# Projekt-Root suchen (wie im Kalibrier-Skript)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CALIBRATION_FILE = Path(PROJECT_ROOT) / "calibration" / "homography.npz"


def main():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(f"Kalibrierdatei nicht gefunden: {CALIBRATION_FILE}")

    data = np.load(str(CALIBRATION_FILE))
    H = data["H"]
    H_inv = data["H_inv"]
    src_pts = data["src_pts"]      # Pixel
    dst_pts = data["dst_pts"]      # Meter
    pitch_length = float(data["pitch_length_m"])
    pitch_width = float(data["pitch_width_m"])
    goal_width = float(data["goal_width_m"])

    print("== Basisinformationen ==")
    print(f"Spielfeld: {pitch_length:.2f}m x {pitch_width:.2f}m, Torbreite: {goal_width:.2f}m")
    print(f"Anzahl Kalibrierpunkte: {len(src_pts)}")
    print()

    # 1) Reprojektionstest: Pixel -> Meter -> Pixel
    print("== Reprojektionstest (Pixel -> Meter -> Pixel) ==")
    errors = []
    for i, (src, dst_world) in enumerate(zip(src_pts, dst_pts)):
        x, y = src
        X, Y = dst_world

        # Pixel -> Meter
        v = np.array([[x, y, 1.0]]).T
        w = H @ v
        X_hat = w[0, 0] / w[2, 0]
        Y_hat = w[1, 0] / w[2, 0]

        # Meter -> Pixel
        v2 = np.array([[X, Y, 1.0]]).T
        w2 = H_inv @ v2
        x_hat = w2[0, 0] / w2[2, 0]
        y_hat = w2[1, 0] / w2[2, 0]

        err_px = float(np.hypot(x_hat - x, y_hat - y))
        err_m = float(np.hypot(X_hat - X, Y_hat - Y))
        errors.append(err_px)

        print(
            f"Punkt {i+1:2d}: "
            f"Pixel=({x:.1f},{y:.1f}) -> Meter_hat=({X_hat:.2f},{Y_hat:.2f}) "
            f" | Pixel-Repro-Error={err_px:.2f}px, Meter-Error={err_m:.3f}m"
        )

    print()
    print(f"Mittlerer Pixel-Repro-Fehler: {np.mean(errors):.2f}px")
    print(f"Maximaler Pixel-Repro-Fehler: {np.max(errors):.2f}px")
    print()

    # 2) Distanz-Test: z.B. Torbreite aus Weltkoordinaten
    print("== Distanz-Checks in Weltkoordinaten ==")
    # Wir wissen: oberer linker/rechter Pfosten sind Punkte 10 und 11 in CALIB_POINTS
    if len(dst_pts) >= 11:
        left_post = dst_pts[9]
        right_post = dst_pts[10]
        d_goal = float(np.hypot(right_post[0] - left_post[0], right_post[1] - left_post[1]))
        print(f"Berechnete Torbreite aus H: {d_goal:.3f}m (Soll: {goal_width:.3f}m)")
    else:
        print("Nicht genügend Punkte für Torbreiten-Check.")

    print("\nFertig. Wenn die Repro-Fehler < ~2-3 Pixel sind und die Torbreite stimmt, ist die Homographie sehr gut.")


if __name__ == "__main__":
    main()