import math
import cv2
import numpy as np

# ===========================================
#  Pfade anpassen
# ===========================================
VIDEO_FILE   = "input_videos_match/Test/kuesnacht_test_clip2.MP4"              # GoPro-Video
PITCH_IMG    = "project/Computer-Vision-FCH/calibration/fch_fussballfeld.jpg"                # Top-Down-Feld
H_PIPE       = "project/Computer-Vision-FCH/calibration/homography.npz"
CALIB_FILE   = "project/Computer-Vision-FCH/calibration/aio_gopro_calib_approx.npz"       # wird erzeugt
H_FILE       = "project/Computer-Vision-FCH/calibration/aio_homography_cam_to_map.npy"    # wird erzeugt
WARP_OUT     = "project/Computer-Vision-FCH/calibration/aio_warped_frame_to_pitch.png"    # wird erzeugt

# ===========================================
#  Hilfsfunktionen
# ===========================================
def export_homography_for_pipeline():

    H = np.load(H_FILE)
    H_inv = np.linalg.inv(H)

    np.savez(H_PIPE, H=H, H_inv=H_inv)


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video konnte nicht geöffnet werden: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Konnte keinen Frame aus dem Video lesen.")
    return frame

def cam_to_map(H, x, y):
    """Punkt (x,y) aus Kamera-Koordinaten via H in Pitch-Pixel umrechnen."""
    p = np.array([x, y, 1.0], dtype=np.float32)
    p_ = H @ p
    X = p_[0] / p_[2]
    Y = p_[1] / p_[2]
    return float(X), float(Y)

# ===========================================
#  STEP 1: GoPro grob entzerren (Slider)
# ===========================================
def step1_gopro_calibration(frame):
    h, w = frame.shape[:2]
    print("Frame size:", w, "x", h)

    # grobe Kameramatrix über angenommenes FOV
    fov_deg = 120.0   # typisches GoPro-Wide FOV
    fov_rad = math.radians(fov_deg)
    fx = (w / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1]], dtype=np.float32)

    print("Initiale Kameramatrix K =\n", K)

    cv2.namedWindow('UNDISTORT', cv2.WINDOW_NORMAL)

    def update(_=None):
        k1_slider = cv2.getTrackbarPos('k1', 'UNDISTORT')
        k2_slider = cv2.getTrackbarPos('k2', 'UNDISTORT')

        k1 = (k1_slider - 100) / 100.0   # -1.0 .. +1.0
        k2 = (k2_slider - 100) / 100.0

        dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        undist = cv2.undistort(frame, K, dist)

        vis = undist.copy()
        text = f"k1={k1:.3f}, k2={k2:.3f}"
        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('UNDISTORT', vis)

    cv2.createTrackbar('k1', 'UNDISTORT', 100, 200, update)
    cv2.createTrackbar('k2', 'UNDISTORT', 100, 200, update)
    update()

    print("\nSTEP 1 – Entzerrung einstellen")
    print("- Schieberegler k1/k2 ändern, bis die Linien (Seitenauslinie, Mittellinie, Strafraum) möglichst gerade aussehen.")
    print("- 's' drücken, um K und dist zu speichern und weiterzugehen.")
    print("- ESC beendet das Programm.\n")

    k1 = 0.0
    k2 = 0.0
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise SystemExit("Abgebrochen in STEP 1.")
        if key == ord('s'):
            k1_slider = cv2.getTrackbarPos('k1', 'UNDISTORT')
            k2_slider = cv2.getTrackbarPos('k2', 'UNDISTORT')
            k1 = (k1_slider - 100) / 100.0
            k2 = (k2_slider - 100) / 100.0
            break

    cv2.destroyAllWindows()

    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    np.savez(CALIB_FILE, K=K, dist=dist)
    print("STEP 1 fertig. Kalibrierung gespeichert als", CALIB_FILE)
    print("K =\n", K)
    print("dist =", dist)

    undist = cv2.undistort(frame, K, dist)
    return K, dist, undist

# ===========================================
#  STEP 2: Homographie Kamera -> Pitch bestimmen
# ===========================================
def step2_homography(cam_img_undist, pitch_img):
    cam_img = cam_img_undist.copy()
    map_img = pitch_img.copy()

    cam_points = []
    map_points = []
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def cam_mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cam_points.append((x, y))
            cv2.circle(cam_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(cam_img, str(len(cam_points)), (x + 5, y - 5),
                        FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            print(f"CAM-Punkt {len(cam_points)}: ({x}, {y})")

    def map_mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            map_points.append((x, y))
            cv2.circle(map_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(map_img, str(len(map_points)), (x + 5, y - 5),
                        FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"MAP-Punkt {len(map_points)}: ({x}, {y})")

    print("\nSTEP 2 – Homographie bestimmen")
    print("1) Fenster 'CAMERA' – nacheinander Punkte anklicken (Schnittpunkte von Linien, alles auf Rasenniveau).")
    print("   -> Mindestens 8–12 Punkte, lieber mehr.")
    print("   -> Wenn fertig: SPACE.")
    print("2) Fenster 'MAP' – dieselben Punkte in derselben Reihenfolge anklicken.")
    print("   -> Fertig: wieder SPACE.")
    print("3) Danach wird H berechnet.\n")

    # Kamera-Punkte
    cv2.namedWindow("CAMERA", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("CAMERA", cam_mouse_cb)
    while True:
        cv2.imshow("CAMERA", cam_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Abgebrochen in STEP 2 (CAMERA).")
        if key == 32:  # SPACE
            break

    print(f"{len(cam_points)} Kamera-Punkte gewählt.\n")

    # Map-Punkte
    cv2.namedWindow("MAP", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("MAP", map_mouse_cb)
    while True:
        cv2.imshow("MAP", map_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Abgebrochen in STEP 2 (MAP).")
        if key == 32:  # SPACE
            break

    print(f"{len(map_points)} Map-Punkte gewählt.")
    cv2.destroyAllWindows()

    if len(cam_points) != len(map_points):
        raise ValueError("Anzahl Kamera- und Map-Punkte unterschiedlich.")
    if len(cam_points) < 4:
        raise ValueError("Zu wenige Punktpaare. Mindestens 4, besser 8+.")

    cam_pts = np.array(cam_points, dtype=np.float32)
    map_pts = np.array(map_points, dtype=np.float32)

    H, mask = cv2.findHomography(cam_pts, map_pts, cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homographie konnte nicht berechnet werden.")

    np.save(H_FILE, H)
    print("\nSTEP 2 fertig. H gespeichert als", H_FILE)
    print("H =\n", H)

    # Warp zum Check
    mh, mw = pitch_img.shape[:2]
    warped = cv2.warpPerspective(cam_img_undist, H, (mw, mh))
    cv2.imwrite(WARP_OUT, warped)
    print("Gewarpter Frame gespeichert als", WARP_OUT)

    stack = np.hstack((pitch_img, warped))
    cv2.namedWindow("MAP (links) | WARPED CAMERA (rechts)", cv2.WINDOW_NORMAL)
    cv2.imshow("MAP (links) | WARPED CAMERA (rechts)", stack)
    print("Vergleich anzeigen – Taste drücken zum Fortfahren.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H

# ===========================================
#  STEP 3: Test – Spieler anklicken -> Punkt auf Pitch
# ===========================================
def step3_click_test(cam_img_undist, pitch_img, H):
    cam_vis = cam_img_undist.copy()
    pitch_vis = pitch_img.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    idx = 0

    def cam_mouse_cb(event, x, y, flags, param):
        nonlocal idx, cam_vis, pitch_vis
        if event == cv2.EVENT_LBUTTONDOWN:
            idx += 1
            # Kamera markieren
            cv2.circle(cam_vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(cam_vis, str(idx), (x + 5, y - 5),
                        FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            X, Y = cam_to_map(H, x, y)
            print(f"Klick {idx}: Kamera=({x},{y}) -> Pitch=({X:.1f},{Y:.1f})")

            h, w = pitch_vis.shape[:2]
            if 0 <= X < w and 0 <= Y < h:
                cv2.circle(pitch_vis, (int(X), int(Y)), 6, (0, 255, 0), -1)
                cv2.putText(pitch_vis, str(idx), (int(X) + 5, int(Y) - 5),
                            FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                print("  -> Achtung: Punkt außerhalb des Pitch-Bildes.")

    cv2.namedWindow("CAMERA_TEST", cv2.WINDOW_NORMAL)
    cv2.namedWindow("PITCH_TEST", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("CAMERA_TEST", cam_mouse_cb)

    print("\nSTEP 3 – Klick-Test")
    print("- Im Fenster 'CAMERA_TEST' auf Spieler (Fußpunkt) klicken.")
    print("- Der Punkt sollte im Fenster 'PITCH_TEST' an der richtigen Feldposition erscheinen.")
    print("- 'r' = Reset aller Punkte, 'q' oder ESC = Beenden.\n")

    while True:
        cv2.imshow("CAMERA_TEST", cam_vis)
        cv2.imshow("PITCH_TEST", pitch_vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('r'):
            cam_vis = cam_img_undist.copy()
            pitch_vis = pitch_img.copy()
            idx = 0
            print("Punkte zurückgesetzt.")

    cv2.destroyAllWindows()

# ===========================================
#  MAIN
# ===========================================
def main():
    # --- Frame & Pitch laden ---
    frame = get_first_frame(VIDEO_FILE)
    pitch_img = cv2.imread(PITCH_IMG)
    if pitch_img is None:
        raise FileNotFoundError(f"Pitch-Bild nicht gefunden: {PITCH_IMG}")

    # --- STEP 1: Entzerrung ---
    K, dist, cam_undist = step1_gopro_calibration(frame)

    # --- STEP 2: Homographie ---
    H = step2_homography(cam_undist, pitch_img)

    # --- STEP 3: Klick-Test ---
    step3_click_test(cam_undist, pitch_img, H)

    print("\nFertig. Du hast jetzt:")
    print(f"- GoPro-Kalibrierung: {CALIB_FILE}")
    print(f"- Homographie:        {H_FILE}")
    print(f"- Gewarptes Bild:     {WARP_OUT}")

if __name__ == "__main__":
    main()