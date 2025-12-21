import sys
import os
import cv2
import numpy as np
import importlib.util
from pathlib import Path

# Resolve repository root from the script location to make execution independent of the working directory
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]

# Locate the central config.py file and fail fast if the repository layout is unexpected
CONFIG_PATH = REPO_ROOT / "project/Computer-Vision-FCH/config.py"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.py nicht gefunden unter: {CONFIG_PATH}")

# Load config.py via importlib to avoid PYTHONPATH issues and ensure deterministic Settings resolution
spec = importlib.util.spec_from_file_location("config", str(CONFIG_PATH))
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

# Instantiate project settings to retrieve standardized file paths and calibration parameters
Settings = config.Settings
s = Settings()

print("[DEBUG] REPO_ROOT:", REPO_ROOT)
print("[DEBUG] config.py geladen von:", CONFIG_PATH)

# Read key input paths from Settings to keep the script environment-agnostic
VIDEO_FILE   = str(s.paths.input_video)
MAP_IMG_FILE = str(s.paths.pitch_image)
H_FILE       = str(s.paths.homography_npy)
CALIB_FILE   = str(s.paths.calib_file)

# Load homography matrix (camera -> pitch) to support coordinate mapping validation
H = np.load(H_FILE)

# Load pitch image as the target canvas for mapped points
map_img = cv2.imread(MAP_IMG_FILE)
if map_img is None:
    raise FileNotFoundError(f"Pitch-Bild nicht gefunden: {MAP_IMG_FILE}")
map_vis = map_img.copy()

# Load camera calibration (intrinsics + distortion) to undistort frames before applying homography
data = np.load(CALIB_FILE)
K = data["K"]
dist = data["dist"]

# Grab a single frame from the video to use as an interactive calibration reference
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise FileNotFoundError(f"Video konnte nicht geöffnet werden: {VIDEO_FILE}")

ok, cam_img_raw = cap.read()
cap.release()
if not ok or cam_img_raw is None:
    raise RuntimeError("Konnte keinen Frame aus dem Video lesen.")

# Undistort the frame so pixel coordinates align with the homography assumptions
cam_img = cv2.undistort(cam_img_raw, K, dist)

print("Frame-Größe (undist):", cam_img.shape)
print("Pitch-Größe:", map_img.shape)

# Transform a camera pixel coordinate into pitch-image coordinates using the homography matrix
def cam_to_map(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float32)
    p_ = H @ p
    X = p_[0] / p_[2]
    Y = p_[1] / p_[2]
    return float(X), float(Y)

# Maintain UI state for click labeling and consistent drawing style across windows
FONT = cv2.FONT_HERSHEY_SIMPLEX
click_idx = 0

# Handle mouse clicks on the camera view and mirror mapped points onto the pitch view for validation
def cam_mouse_cb(event, x, y, flags, param):
    global cam_img, map_vis, click_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        click_idx += 1

        cv2.circle(cam_img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(cam_img, str(click_idx), (x + 5, y - 5),
                    FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        X, Y = cam_to_map(H, x, y)
        print(f"Klick {click_idx}: Kamera=({x},{y})  ->  Pitch=({X:.1f},{Y:.1f})")

        # Guard against out-of-bounds mappings to highlight calibration/homography issues
        h, w = map_vis.shape[:2]
        if 0 <= X < w and 0 <= Y < h:
            cv2.circle(map_vis, (int(X), int(Y)), 6, (0, 255, 0), -1)
            cv2.putText(map_vis, str(click_idx), (int(X) + 5, int(Y) - 5),
                        FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print("  -> Achtung: gemappter Punkt liegt außerhalb des Pitch-Bildes.")

# Create interactive windows and attach callbacks to support manual mapping QA
cv2.namedWindow("CAMERA", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("CAMERA", cam_mouse_cb)
cv2.namedWindow("PITCH", cv2.WINDOW_NORMAL)

print("Anleitung:")
print("- Im Fenster 'CAMERA' auf einen Spieler (am Fußpunkt) klicken.")
print("- Jeder Klick wird live im Fenster 'PITCH' als Punkt angezeigt.")
print("- 'r' drücken, um alle Punkte zu resetten.")
print("- 'ESC' oder 'q' zum Beenden.\n")

# Keep pristine copies to enable fast resets without reloading assets
cam_orig = cam_img.copy()

# Run the OpenCV UI loop to display views, capture interactions, and handle reset/exit controls
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

# Cleanly release OpenCV resources to avoid hanging windows in subsequent runs
cv2.destroyAllWindows()