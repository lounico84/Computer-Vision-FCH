import cv2
import numpy as np

# ==== Pfade anpassen ====
VIDEO_FILE   = "/Users/loubrauchli/Documents/Python_Projects/Project_Football_Analytics_FCHERRLIBERG/input_videos_match/Test/kuesnacht_test_clip2.MP4"
MAP_IMG_FILE = "project/Computer-Vision-FCH/calibration/fch_fussballfeld.jpg"
H_FILE       = "project/Computer-Vision-FCH/calibration/aio_homography_cam_to_map.npy"
CALIB_FILE   = "project/Computer-Vision-FCH/calibration/aio_gopro_calib_approx.npz"  # dein approx-Kalib-File

# ==== 1. Daten laden ====
# Homographie
H = np.load(H_FILE)

# Pitch-Bild
map_img = cv2.imread(MAP_IMG_FILE)
if map_img is None:
    raise FileNotFoundError(f"Pitch-Bild nicht gefunden: {MAP_IMG_FILE}")
map_vis = map_img.copy()

# Kalibrierung laden
data = np.load(CALIB_FILE)
K = data["K"]
dist = data["dist"]

# Video und ersten Frame holen
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise FileNotFoundError(f"Video konnte nicht geöffnet werden: {VIDEO_FILE}")

ok, cam_img_raw = cap.read()
cap.release()
if not ok or cam_img_raw is None:
    raise RuntimeError("Konnte keinen Frame aus dem Video lesen.")

# ENTZERRTER Frame (muss zur H passen!)
cam_img = cv2.undistort(cam_img_raw, K, dist)

print("Frame-Größe (undist):", cam_img.shape)
print("Pitch-Größe:", map_img.shape)

# ==== 2. Hilfsfunktion: Punkt mit H transformieren ====
def cam_to_map(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float32)
    p_ = H @ p
    X = p_[0] / p_[2]
    Y = p_[1] / p_[2]
    return float(X), float(Y)

# ==== 3. Maus-Callback ====
FONT = cv2.FONT_HERSHEY_SIMPLEX
click_idx = 0

def cam_mouse_cb(event, x, y, flags, param):
    global cam_img, map_vis, click_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        click_idx += 1

        cv2.circle(cam_img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(cam_img, str(click_idx), (x + 5, y - 5),
                    FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        X, Y = cam_to_map(H, x, y)
        print(f"Klick {click_idx}: Kamera=({x},{y})  ->  Pitch=({X:.1f},{Y:.1f})")

        h, w = map_vis.shape[:2]
        if 0 <= X < w and 0 <= Y < h:
            cv2.circle(map_vis, (int(X), int(Y)), 6, (0, 255, 0), -1)
            cv2.putText(map_vis, str(click_idx), (int(X) + 5, int(Y) - 5),
                        FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print("  -> Achtung: gemappter Punkt liegt außerhalb des Pitch-Bildes.")

# ==== 4. Fenster einrichten ====
cv2.namedWindow("CAMERA", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("CAMERA", cam_mouse_cb)
cv2.namedWindow("PITCH", cv2.WINDOW_NORMAL)

print("Anleitung:")
print("- Im Fenster 'CAMERA' auf einen Spieler (am Fußpunkt) klicken.")
print("- Jeder Klick wird live im Fenster 'PITCH' als Punkt angezeigt.")
print("- 'r' drücken, um alle Punkte zu resetten.")
print("- 'ESC' oder 'q' zum Beenden.\n")

cam_orig = cam_img.copy()

# ==== 5. Event-Loop ====
while True:
    cv2.imshow("CAMERA", cam_img)
    cv2.imshow("PITCH", map_vis)

    key = cv2.waitKey(20) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('r'):
        click_idx = 0
        map_vis = map_img.copy()
        cam_img = cam_orig.copy()
        print("Punkte zurückgesetzt.")

cv2.destroyAllWindows()