import math
import cv2
import numpy as np
import sys
import os
import pathlib as Path
from config import Settings

# Einstellungen laden
s = Settings()
analytics_cfg = s.analytics

# ===========================================
#  PFADE (aus config.py)
# ===========================================
VIDEO_FILE     = str(s.paths.input_video)           # Das Match-Video
PITCH_IMG_PATH = str(s.paths.pitch_image)           # Das 2D-Spielfeld-Bild
CALIB_FILE     = str(s.paths.calib_file)            # Die zentrale GoPro-Entzerrung
H_PIXEL_FILE   = str(s.paths.homography_npy)        # Zwischenspeicher (Pixel-Homographie)
H_METER_FILE   = str(s.paths.homography_npz)        # FINAL: Meter-Homographie für die Pipeline
WARP_OUT       = str(s.paths.warped_frame_output)   # Debug-Bild

# Echte Feldgröße aus Config (für die Meter-Umrechnung)
FIELD_LENGTH_M = analytics_cfg.pitch_length
FIELD_WIDTH_M  = analytics_cfg.pitch_width

# ===========================================
#  HILFSFUNKTION: Pixel -> Meter Konvertierung
# ===========================================
def export_homography_to_meters(H_pixel, pitch_img_path, out_npz_path):
    """
    Nimmt die geklickte Pixel-Homographie, berechnet die Skalierung auf Meter
    basierend auf der Bildgröße des Pitch-Images und der echten Feldgröße.
    """
    img = cv2.imread(pitch_img_path)
    if img is None:
        raise FileNotFoundError(f"Pitch-Bild nicht gefunden: {pitch_img_path}")
    
    h_img, w_img = img.shape[:2]
    print(f"\n[Export] Pitch-Bildgröße: {w_img} x {h_img} px")
    print(f"[Export] Reale Feldgröße: {FIELD_LENGTH_M} x {FIELD_WIDTH_M} m")

    # Skalierungs-Matrix S: Transformiert Pitch-Pixel in echte Meter
    # Formel: x_meter = (x_pixel / bild_breite) * feld_länge
    S = np.array([
        [FIELD_LENGTH_M / w_img, 0.0,                     0.0],
        [0.0,                    FIELD_WIDTH_M / h_img,   0.0],
        [0.0,                    0.0,                     1.0]
    ], dtype=np.float32)

    # Die finale Homographie (Kamera -> Meter) ist S * H_pixel
    H_meter = S @ H_pixel
    
    # Inverse berechnen (Meter -> Kamera)
    try:
        H_inv_meter = np.linalg.inv(H_meter)
    except np.linalg.LinAlgError:
        print("[ERROR] Homographie ist singulär und nicht invertierbar!")
        return

    # Speichern für die Pipeline
    np.savez(out_npz_path, H=H_meter, H_inv=H_inv_meter)
    
    print(f"[Export] SUCCESS! Meter-Homographie gespeichert in: {out_npz_path}")
    print("H_meter Matrix:\n", H_meter)


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video konnte nicht geöffnet werden: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Konnte keinen Frame aus dem Video lesen.")
    return frame

# ===========================================
#  STEP 1: GoPro Entzerrung (Laden oder Erstellen)
# ===========================================
def step1_gopro_calibration(frame):
    print("\n--- STEP 1: Linsen-Entzerrung ---")
    
    # Prüfen, ob die zentrale Kalibrierungsdatei existiert
    if os.path.exists(CALIB_FILE):
        print(f"[INFO] Zentrale Kalibrierung gefunden: {CALIB_FILE}")
        try:
            data = np.load(CALIB_FILE)
            K = data['K']
            dist = data['dist']
            
            # Kurzer Check auf Nullen
            if np.all(dist == 0):
                print("[WARNUNG] Datei enthält nur Nullen! Starte manuelle Kalibrierung...")
            else:
                undist = cv2.undistort(frame, K, dist)
                print("-> Automatisch geladen (Gleiche Kamera). Überspringe Slider.")
                return K, dist, undist
        except Exception as e:
            print(f"[ERROR] Fehler beim Laden: {e}. Starte manuelle Kalibrierung...")

    # Falls nicht vorhanden oder fehlerhaft -> Manuell erstellen
    print("-> Starte manuelle Kalibrierung (Slider)...")
    h, w = frame.shape[:2]
    
    # Initiale Matrix
    fov_deg = 120.0
    fov_rad = math.radians(fov_deg)
    fx = (w / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    cv2.namedWindow('UNDISTORT (Druecke S zum Speichern)', cv2.WINDOW_NORMAL)

    def update(_=None):
        k1 = (cv2.getTrackbarPos('k1', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
        k2 = (cv2.getTrackbarPos('k2', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
        dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        vis = cv2.undistort(frame, K, dist)
        
        # Info-Text
        cv2.putText(vis, f"k1={k1:.2f}, k2={k2:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('UNDISTORT (Druecke S zum Speichern)', vis)

    cv2.createTrackbar('k1', 'UNDISTORT (Druecke S zum Speichern)', 100, 200, update)
    cv2.createTrackbar('k2', 'UNDISTORT (Druecke S zum Speichern)', 100, 200, update)
    update()

    print("Stelle die Slider ein, bis Linien gerade sind.")
    print("Drücke 's', um zu speichern und fortzufahren.")
    
    while True:
        if cv2.waitKey(20) & 0xFF == ord('s'):
            break
    cv2.destroyAllWindows()

    # Werte auslesen
    k1 = (cv2.getTrackbarPos('k1', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
    k2 = (cv2.getTrackbarPos('k2', 'UNDISTORT (Druecke S zum Speichern)') - 100) / 100.0
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    # Speichern
    np.savez(CALIB_FILE, K=K, dist=dist)
    print(f"Kalibrierung gespeichert unter: {CALIB_FILE}")
    
    return K, dist, cv2.undistort(frame, K, dist)

# ===========================================
#  STEP 2: Homographie (Punkte klicken)
# ===========================================
def step2_homography(cam_img_undist, pitch_img):
    print("\n--- STEP 2: Homographie (Punkte klicken) ---")
    print("1. Klicke Punkt im VIDEO (z.B. Eckfahne)")
    print("2. Klicke denselben Punkt auf der MAP")
    print("-> Mindestens 4 Paare. SPACE zum Berechnen.")
    
    cam_display = cam_img_undist.copy()
    map_display = pitch_img.copy()
    cam_pts = []
    map_pts = []

    def cam_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cam_pts.append((x, y))
            cv2.circle(cam_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(cam_display, str(len(cam_pts)), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("CAMERA", cam_display)

    def map_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            map_pts.append((x, y))
            cv2.circle(map_display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(map_display, str(len(map_pts)), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("MAP", map_display)

    cv2.namedWindow("CAMERA", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("CAMERA", cam_cb)
    cv2.imshow("CAMERA", cam_display)

    cv2.namedWindow("MAP", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("MAP", map_cb)
    cv2.imshow("MAP", map_display)

    while True:
        if cv2.waitKey(20) & 0xFF == 32: # Space Taste
            break
    cv2.destroyAllWindows()

    if len(cam_pts) < 4:
        raise ValueError("Zu wenige Punkte! Mindestens 4 notwendig.")
    if len(cam_pts) != len(map_pts):
        raise ValueError(f"Ungleiche Anzahl Punkte! Video: {len(cam_pts)}, Map: {len(map_pts)}")

    # Homographie berechnen (Pixel -> Pixel)
    H, _ = cv2.findHomography(np.array(cam_pts), np.array(map_pts))
    
    # Speichern der RAW-Matrix (optional, falls man später manuell umrechnen will)
    np.save(H_PIXEL_FILE, H)
    
    # Warp Check (Visuelle Prüfung)
    h_map, w_map = pitch_img.shape[:2]
    warped = cv2.warpPerspective(cam_img_undist, H, (w_map, h_map))
    cv2.imwrite(WARP_OUT, warped)
    print(f"Visueller Check gespeichert: {WARP_OUT}")
    
    return H

# ===========================================
#  MAIN
# ===========================================
def main():
    # 1. Daten laden
    frame = get_first_frame(VIDEO_FILE)
    pitch_img = cv2.imread(PITCH_IMG_PATH)
    if pitch_img is None:
        raise FileNotFoundError(f"Pitch-Bild fehlt: {PITCH_IMG_PATH}")

    # 2. Schritt 1: Entzerrung (automatisch laden, wenn vorhanden)
    K, dist, cam_undist = step1_gopro_calibration(frame)

    # 3. Schritt 2: Punkte klicken (Pixel -> MapPixel)
    # Hier musst du für jedes Match NEU klicken!
    H_pixel = step2_homography(cam_undist, pitch_img)

    # 4. Schritt 3: Automatisch in METER umrechnen und speichern
    print("\n--- STEP 3: Exportiere in Meter... ---")
    export_homography_to_meters(H_pixel, PITCH_IMG_PATH, H_METER_FILE)

    print("\n-----------------------------------------------------------")
    print("[FERTIG] Kalibrierung erfolgreich abgeschlossen.")
    print(f"1. Entzerrung geladen von: {CALIB_FILE}")
    print(f"2. Homographie (Meter) gespeichert in: {H_METER_FILE}")
    print("-----------------------------------------------------------")
    print("Du kannst jetzt main.py starten (stelle sicher, dass read_tracks_from_stub=False ist).")

if __name__ == "__main__":
    main()