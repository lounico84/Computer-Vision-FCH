# calibration/test_homography_overlay.py

import os
import sys
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Settings  # noqa: E402
from utils import pitch_to_pixel  # aus homography_utils.py


def _get_first_frame():
    settings = Settings()
    video_path = str(settings.paths.input_video)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Video nicht öffnen: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Konnte kein Frame lesen.")
    return frame


def draw_pitch_overlay(frame, pitch_length=105.0, pitch_width=68.0):
    """
    Zeichne einige wichtige Linien des Spielfelds ins Bild:
    - Außenlinien
    - Mittellinie
    - Mittelkreis (als Polygon)
    """
    img = frame.copy()

    # 1) Außenlinien (Rechteck)
    corners_world = [
        (0.0, 0.0),
        (pitch_length, 0.0),
        (pitch_length, pitch_width),
        (0.0, pitch_width),
    ]
    corners_px = [pitch_to_pixel(X, Y) for (X, Y) in corners_world]
    for i in range(len(corners_px)):
        x1, y1 = corners_px[i]
        x2, y2 = corners_px[(i + 1) % len(corners_px)]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 2) Mittellinie
    mid_y = pitch_width / 2.0
    m1 = pitch_to_pixel(0.0, mid_y)
    m2 = pitch_to_pixel(pitch_length, mid_y)
    cv2.line(img, (int(m1[0]), int(m1[1])), (int(m2[0]), int(m2[1])), (255, 0, 0), 2)

    # 3) Mittelkreis (als viele kurze Linien)
    center_x = pitch_length / 2.0
    center_y = pitch_width / 2.0
    radius = 9.15
    num_segments = 72
    circle_pts = []
    for k in range(num_segments + 1):
        angle = 2 * np.pi * k / num_segments
        X = center_x + radius * np.cos(angle)
        Y = center_y + radius * np.sin(angle)
        circle_pts.append(pitch_to_pixel(X, Y))

    for i in range(len(circle_pts) - 1):
        x1, y1 = circle_pts[i]
        x2, y2 = circle_pts[i + 1]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    return img


def main():
    frame = _get_first_frame()
    overlay = draw_pitch_overlay(frame)

    cv2.namedWindow("Homography Overlay", cv2.WINDOW_NORMAL)
    cv2.imshow("Homography Overlay", overlay)
    print("Fenster zeigt Overlay. Drücke eine Taste, um zu schließen.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_path = Path(PROJECT_ROOT) / "calibration" / "homography_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"Overlay-Bild gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()