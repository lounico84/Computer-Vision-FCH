# calibration/homography_calibration.py
#
# Multi-Point Homographie-Kalibrierung für dein Spielfeld.
#
# Idee:
# - Du klickst eine Reihe von definierten Spielfeldpunkten im Bild an
#   (Mittellinie, Mittelkreis, obere Ecken, Torpfosten).
# - Für jeden dieser Bildpunkte kennen wir die realen Feldkoordinaten in Metern.
# - Mit allen diesen Punktpaaren berechnen wir eine robuste Homographie H
#   (Pixel -> Meter) via cv2.findHomography(..., RANSAC).
# - Das Ergebnis wird als calibration/homography.npz gespeichert und kann
#   später von utils/homography_utils.py geladen und benutzt werden.

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------
# Projekt-Root zum Python-Suchpfad hinzufügen,
# damit `from config import Settings` funktioniert
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Settings  # noqa: E402


# ---------------------------------------------------------
# Spielfeld-Parameter (kannst du bei Bedarf anpassen)
# ---------------------------------------------------------
PITCH_LENGTH_M = 105.0  # Länge (x-Richtung, von linker zu rechter Seitenlinie)
PITCH_WIDTH_M = 68.0    # Breite (y-Richtung, von oberer zu unterer Torlinie)
GOAL_WIDTH_M = 7.32     # Torbreite (Pfosten zu Pfosten)

CENTER_X = PITCH_LENGTH_M / 2.0  # 52.5 m
CENTER_Y = PITCH_WIDTH_M / 2.0   # 34.0 m

MIDL_CIRCLE_RADIUS = 9.15        # Radius des Mittelkreises
HALF_GOAL = GOAL_WIDTH_M / 2.0   # 3.66 m

# Datei, in der die Homographie gespeichert wird (relativ zum Projekt-Root)
CALIBRATION_FILE = "calibration/homography.npz"


# ---------------------------------------------------------
# Kalibrierpunkte
#
# Jeder Eintrag:
#   ("Beschreibung für Overlay", (X_in_Metern, Y_in_Metern))
#
# Die REIHENFOLGE in dieser Liste ist GENAU die Reihenfolge,
# in der du im Bild klickst.
#
# Punkte 1–7 entsprechen deiner markierten Skizze,
# Punkte 8–11 ergänzen obere Ecken und Torpfosten für Stabilität.
# ---------------------------------------------------------
CALIB_POINTS = [
    # --- DEINE PUNKTE 1–7 (siehe Screenshot) ---

    # 1) Unteres Ende der Mittellinie (Schnitt mit unterer Tor-/Seitenlinie)
    ("Unteres Ende Mittellinie", (CENTER_X, PITCH_WIDTH_M)),                        # (52.5, 68.0)

    # 2) Untere Mittelkreis-Kante (Mittellinie trifft unteren Teil des Mittelkreises)
    ("Untere Mittelkreis-Kante", (CENTER_X, CENTER_Y + MIDL_CIRCLE_RADIUS)),        # (52.5, 43.15)

    # 3) Mittelpunkt Spielfeld / Anstoßpunkt
    ("Mittelpunkt Spielfeld", (CENTER_X, CENTER_Y)),                                # (52.5, 34.0)

    # 4) Oberes Ende der Mittellinie (Schnitt mit oberer Torlinie)
    ("Oberes Ende Mittellinie", (CENTER_X, 0.0)),                                   # (52.5, 0.0)

    # 5) Obere Mittelkreis-Kante (Mittellinie trifft oberen Teil des Mittelkreises)
    ("Obere Mittelkreis-Kante", (CENTER_X, CENTER_Y - MIDL_CIRCLE_RADIUS)),         # (52.5, 24.85)

    # 6) Linke Mittelkreis-Kante
    ("Linke Mittelkreis-Kante", (CENTER_X - MIDL_CIRCLE_RADIUS, CENTER_Y)),         # (43.35, 34.0)

    # 7) Rechte Mittelkreis-Kante
    ("Rechte Mittelkreis-Kante", (CENTER_X + MIDL_CIRCLE_RADIUS, CENTER_Y)),        # (61.65, 34.0)

    # --- ZUSÄTZLICHE STABILE PUNKTE ---

    # 8) Obere linke Ecke des Spielfelds
    ("Obere linke Ecke", (0.0, 0.0)),

    # 9) Obere rechte Ecke des Spielfelds
    ("Obere rechte Ecke", (PITCH_LENGTH_M, 0.0)),                                   # (105.0, 0.0)

    # 10) Oberer linker Torpfosten
    ("Oberer linker Torpfosten", (CENTER_X - HALF_GOAL, 0.0)),                      # (48.84, 0.0)

    # 11) Oberer rechter Torpfosten
    ("Oberer rechter Torpfosten", (CENTER_X + HALF_GOAL, 0.0)),                     # (56.16, 0.0)
]


def _get_first_frame():
    """
    Liest das erste Frame des Input-Videos, das in config.Settings.paths.input_video
    konfiguriert ist.
    """
    settings = Settings()
    video_path = str(settings.paths.input_video)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Video nicht öffnen: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Konnte kein Frame aus dem Video lesen.")
    return frame


def run_calibration():
    """
    Hauptfunktion:
    - Zeigt das erste Videoframe an.
    - Du klickst nacheinander alle Punkte aus CALIB_POINTS.
    - Danach wird die Homographie H berechnet und gespeichert.
    """
    frame = _get_first_frame()
    base_frame = frame.copy()

    window_name = "Homographie-Kalibrierung (Multi-Point)"
    cv2.namedWindow(window_name)

    clicked_points = []

    num_points = len(CALIB_POINTS)

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < num_points:
                clicked_points.append((x, y))
                label, _ = CALIB_POINTS[len(clicked_points) - 1]
                print(
                    f"Punkt {len(clicked_points)}/{num_points}: "
                    f"{label} -> Pixel ({x}, {y})"
                )

    cv2.setMouseCallback(window_name, mouse_callback)

    # Anleitungs-Text: Kopf + Punktliste + Footer
    instructions_header = [
        "Klicke die folgenden Spielfeldpunkte in dieser Reihenfolge:",
    ]
    instructions_points = [
        f"{i + 1}) {desc}" for i, (desc, _) in enumerate(CALIB_POINTS)
    ]
    instructions_footer = [
        "",
        "Druecke 'r' zum Reset, 'q' zum Abbrechen.",
    ]

    while True:
        display = base_frame.copy()

        # Text oben links zeichnen
        y0 = 20
        for i, line in enumerate(
            instructions_header + instructions_points + instructions_footer
        ):
            cv2.putText(
                display,
                line,
                (20, y0 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # bereits geklickte Punkte markieren
        for idx, (x, y) in enumerate(clicked_points):
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                display,
                str(idx + 1),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            print("Kalibrierung abgebrochen.")
            cv2.destroyAllWindows()
            return

        if key == ord("r"):
            print("Punkte zurueckgesetzt.")
            clicked_points = []

        if len(clicked_points) == num_points:
            break

    cv2.destroyAllWindows()

    # -----------------------------------------------------
    # Homographie aus allen Punktpaaren berechnen
    # -----------------------------------------------------
    src_pts = np.array(clicked_points, dtype=np.float32)             # Pixelpunkte
    dst_pts = np.array([pt for _, pt in CALIB_POINTS], dtype=np.float32)  # Feldkoords (Meter)

    # RANSAC macht die Schätzung robust gegenüber Ausreißern / Klickfehlern.
    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)

    if H is None:
        raise RuntimeError("cv2.findHomography hat keine gültige Matrix geliefert.")

    H_inv = np.linalg.inv(H)

    calib_path = Path(PROJECT_ROOT) / CALIBRATION_FILE
    calib_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(calib_path),
        H=H,
        H_inv=H_inv,
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
        goal_width_m=GOAL_WIDTH_M,
        src_pts=src_pts,
        dst_pts=dst_pts,
        mask=mask,
    )

    print(f"\nHomographie gespeichert unter: {calib_path}")
    print("H =\n", H)
    print("\nH_inv =\n", H_inv)
    print("\nVerwendete Punkte (Pixel -> Meter):")
    for (desc, world), (px, py) in zip(CALIB_POINTS, src_pts):
        print(
            f"  {desc:30s}  Pixel=({px:.1f}, {py:.1f})"
            f" -> Meter=({world[0]:.2f}, {world[1]:.2f})"
        )


if __name__ == "__main__":
    run_calibration()