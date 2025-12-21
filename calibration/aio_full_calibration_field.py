import math
import cv2
import numpy as np
import sys
import os
import pathlib as Path
from config import Settings

# Load project settings to centralize analytics parameters and file paths
s = Settings()
analytics_cfg = s.analytics

# Define key inputs/outputs from config to keep calibration workflow reproducible
VIDEO_FILE     = str(s.paths.input_video)           # Match video input
PITCH_IMG_PATH = str(s.paths.pitch_image)           # 2D pitch reference image
CALIB_FILE     = str(s.paths.calib_file)            # Shared GoPro undistortion parameters
H_PIXEL_FILE   = str(s.paths.homography_npy)        # Intermediate pixel-space homography
H_METER_FILE   = str(s.paths.homography_npz)        # Final metric homography used by the pipeline
WARP_OUT       = str(s.paths.warped_frame_output)   # Debug warp output for visual QA

# Cache real-world pitch dimensions (meters) for pixel-to-metric conversion
FIELD_LENGTH_M = analytics_cfg.pitch_length
FIELD_WIDTH_M  = analytics_cfg.pitch_width


# Convert a pixel-space homography (camera -> pitch image) into a meter-space homography (camera -> real pitch)
def export_homography_to_meters(H_pixel, pitch_img_path, out_npz_path):
    # Load pitch image to derive scaling factors between pitch pixels and real meters
    img = cv2.imread(pitch_img_path)
    if img is None:
        raise FileNotFoundError(f"Pitch-Bild nicht gefunden: {pitch_img_path}")
    
    h_img, w_img = img.shape[:2]
    print(f"\n[Export] Pitch-Bildgröße: {w_img} x {h_img} px")
    print(f"[Export] Reale Feldgröße: {FIELD_LENGTH_M} x {FIELD_WIDTH_M} m")

    # Build a scale matrix mapping pitch-image pixels to metric pitch coordinates
    S = np.array([
        [FIELD_LENGTH_M / w_img, 0.0,                     0.0],
        [0.0,                    FIELD_WIDTH_M / h_img,   0.0],
        [0.0,                    0.0,                     1.0]
    ], dtype=np.float32)

    # Compose final homography in meters to align downstream analytics with real units
    H_meter = S @ H_pixel
    
    # Compute inverse mapping for optional back-projection and diagnostics
    try:
        H_inv_meter = np.linalg.inv(H_meter)
    except np.linalg.LinAlgError:
        print("[ERROR] Homography is singular and cannot be inverted.")
        return

    # Persist both forward and inverse transforms for pipeline consumption
    np.savez(out_npz_path, H=H_meter, H_inv=H_inv_meter)
    
    print(f"[Export] SUCCESS! Meter-Homographie gespeichert in: {out_npz_path}")
    print("H_meter Matrix:\n", H_meter)


# Read a single representative video frame to use for manual calibration and homography selection
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video konnte nicht geöffnet werden: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Konnte keinen Frame aus dem Video lesen.")
    return frame


# Undistort the camera frame using stored calibration, or provide a manual slider-based fallback
def step1_gopro_calibration(frame):
    print("\n--- STEP 1: Linsen-Entzerrung ---")
    
    # Reuse existing calibration when available to keep per-match setup minimal
    if os.path.exists(CALIB_FILE):
        print(f"[INFO] Zentrale Kalibrierung gefunden: {CALIB_FILE}")
        try:
            data = np.load(CALIB_FILE)
            K = data['K']
            dist = data['dist']
            
            # Detect invalid placeholder calibration and force manual setup when needed
            if np.all(dist == 0):
                print("[WARNUNG] Datei enthält nur Nullen! Starte manuelle Kalibrierung...")
            else:
                undist = cv2.undistort(frame, K, dist)
                print("-> Automatisch geladen (Gleiche Kamera). Überspringe Slider.")
                return K, dist, undist
        except Exception as e:
            print(f"[ERROR] Fehler beim Laden: {e}. Starte manuelle Kalibrierung...")

    # Provide an interactive calibration path to tune distortion parameters on-the-fly
    print("-> Starte manuelle Kalibrierung (Slider)...")
    h, w = frame.shape[:2]
    
    # Initialize intrinsics from an assumed GoPro field-of-view to stabilize slider tuning
    fov_deg = 120.0
    fov_rad = math.radians(fov_deg)
    fx = (w / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    cv2.namedWindow('UNDISTORT (Druecke S zum Speichern)', cv2.WINDOW_NORMAL)

    # Update preview on slider movement to visually converge towards straight lines
    def update(_=None):
        k1 = (cv2.getTrackbarPos('k1', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
        k2 = (cv2.getTrackbarPos('k2', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
        dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        vis = cv2.undistort(frame, K, dist)
        
        cv2.putText(vis, f"k1={k1:.2f}, k2={k2:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('UNDISTORT (Druecke S zum Speichern)', vis)

    cv2.createTrackbar('k1', 'UNDISTORT (Druecke S zum Speichern)', 100, 200, update)
    cv2.createTrackbar('k2', 'UNDISTORT (Druecke S zum Speichern)', 100, 200, update)
    update()

    print("Stelle die Slider ein, bis Linien gerade sind.")
    print("Drücke 's', um zu speichern und fortzufahren.")
    
    # Block until the user confirms the calibration snapshot
    while True:
        if cv2.waitKey(20) & 0xFF == ord('s'):
            break
    cv2.destroyAllWindows()

    # Persist selected distortion coefficients to reuse across matches with the same camera setup
    k1 = (cv2.getTrackbarPos('k1', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
    k2 = (cv2.getTrackbarPos('k2', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    np.savez(CALIB_FILE, K=K, dist=dist)
    print(f"Kalibrierung gespeichert unter: {CALIB_FILE}")
    
    return K, dist, cv2.undistort(frame, K, dist)


# Collect corresponding points (camera frame vs pitch image) and compute a camera->pitch homography
def step2_homography(cam_img_undist, pitch_img):
    print("\n--- STEP 2: Homographie (Punkte klicken) ---")
    print("1. Klicke Punkt im VIDEO (z.B. Eckfahne)")
    print("2. Klicke denselben Punkt auf der MAP")
    print("-> Mindestens 4 Paare. SPACE zum Berechnen.")
    
    cam_display = cam_img_undist.copy()
    map_display = pitch_img.copy()
    cam_pts = []
    map_pts = []

    # Capture camera-frame points for homography estimation with visual numbering feedback
    def cam_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cam_pts.append((x, y))
            cv2.circle(cam_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(cam_display, str(len(cam_pts)), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("CAMERA", cam_display)

    # Capture pitch-image points for homography estimation with visual numbering feedback
    def map_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            map_pts.append((x, y))
            cv2.circle(map_display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(map_display, str(len(map_pts)), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("MAP", map_display)

    # Create interactive windows to collect point pairs from both coordinate systems
    cv2.namedWindow("CAMERA", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("CAMERA", cam_cb)
    cv2.imshow("CAMERA", cam_display)

    cv2.namedWindow("MAP", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("MAP", map_cb)
    cv2.imshow("MAP", map_display)

    # Wait for user confirmation (space) to compute homography from collected correspondences
    while True:
        if cv2.waitKey(20) & 0xFF == 32:
            break
    cv2.destroyAllWindows()

    # Validate minimum correspondence count and one-to-one matching before estimation
    if len(cam_pts) < 4:
        raise ValueError("Zu wenige Punkte! Mindestens 4 notwendig.")
    if len(cam_pts) != len(map_pts):
        raise ValueError(f"Ungleiche Anzahl Punkte! Video: {len(cam_pts)}, Map: {len(map_pts)}")

    # Estimate pixel-space homography (camera pixels -> pitch-image pixels)
    H, _ = cv2.findHomography(np.array(cam_pts), np.array(map_pts))
    
    # Persist the pixel-space homography for debugging and repeatability
    np.save(H_PIXEL_FILE, H)
    
    # Generate a warp preview to visually QA alignment before exporting meter-space transforms
    h_map, w_map = pitch_img.shape[:2]
    warped = cv2.warpPerspective(cam_img_undist, H, (w_map, h_map))
    cv2.imwrite(WARP_OUT, warped)
    print(f"Visueller Check gespeichert: {WARP_OUT}")
    
    return H


# Run the end-to-end calibration pipeline: undistort, click correspondences, export meter homography for production
def main():
    # Load reference assets (frame + pitch image) to support calibration and QA
    frame = get_first_frame(VIDEO_FILE)
    pitch_img = cv2.imread(PITCH_IMG_PATH)
    if pitch_img is None:
        raise FileNotFoundError(f"Pitch-Bild fehlt: {PITCH_IMG_PATH}")

    # Step 1: undistort using stored calibration when available (otherwise manual tuning)
    K, dist, cam_undist = step1_gopro_calibration(frame)

    # Step 2: collect point pairs and compute camera->pitch pixel-space homography
    H_pixel = step2_homography(cam_undist, pitch_img)

    # Step 3: convert and persist a meter-space homography for downstream analytics modules
    print("\n--- STEP 3: Exportiere in Meter... ---")
    export_homography_to_meters(H_pixel, PITCH_IMG_PATH, H_METER_FILE)

    print("\n-----------------------------------------------------------")
    print("[FERTIG] Kalibrierung erfolgreich abgeschlossen.")
    print(f"1. Entzerrung geladen von: {CALIB_FILE}")
    print(f"2. Homographie (Meter) gespeichert in: {H_METER_FILE}")
    print("-----------------------------------------------------------")
    print("Du kannst jetzt main.py starten (stelle sicher, dass read_tracks_from_stub=False ist).")


# Execute calibration only when run as a script to keep the module import-safe for pipelines
if __name__ == "__main__":
    main()