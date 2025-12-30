# Computer Vision Fussballanalyse

End-to-End Computer-Vision-Pipeline zur Analyse von Fussballspielen auf Amateur- und semiprofessionellem Niveau.  
Das System erkennt und verfolgt Spieler, Torhüter, Schiedsrichter und den Ball, weist Teams anhand der Trikotfarben zu, projiziert Kamerapixel mittels Homographie in Spielfeldkoordinaten (Meter), berechnet frame-basierte Events und erzeugt visuelle Overlays, Analysen sowie einen automatisierten PDF-Report.

---

## Hauptfunktionen

- **Objekterkennung & Tracking**
  - YOLO (Ultralytics) für Detektion
  - ByteTrack (Supervision) für stabile Track-IDs über Zeit

- **Ball-Trajektorien**
  - Interpolation fehlender Ballpositionen
  - Filterung unplausibler Sprünge

- **Team-Zuordnung**
  - Trikotfarben-Extraktion mit HSV-basiertem Rasenfilter
  - Robuste Bestimmung der Teamfarben aus den ersten Frames
  - „Sticky“-Logik zur Stabilisierung unsicherer Frames

- **Spielfeld-Transformation**
  - Homographie von Kamerapixeln zu Spielfeld-Metern
  - Export von Vorwärts- und Invers-Homographie (`H`, `H_inv`)
  - Lazy Loading über Utility-Funktionen

- **Ballbesitz-Erkennung**
  - Zuordnung Ball → Spieler über Fuss-Punkt in Meterkoordinaten
  - Hysterese für Besitzer- und Teamwechsel
  - Unterstützung für kurze „Freier-Ball“-Phasen (Pässe, Chipbälle)

- **Analysen & Visualisierungen**
  - Pass-Maps, Heatmaps, Zonenanalyse
  - Match Momentum (rollender Ballbesitz)
  - Shot Map & einfache Torerkennung
  - Live-Mini-Map mit Voronoi-Raumkontrolle

- **Reporting**
  - Export eines frame-basierten CSVs als zentrale Datenschnittstelle
  - Automatische Generierung eines PDF-Reports aus einem Jupyter Notebook

---

## Projektstruktur

Typische Module im Repository:

- `pipeline/`
  - `io_pipeline.py` – Video laden und Tracking generieren/laden
  - `team_pipeline.py` – Team- und Torhüter-Zuordnung
  - `ball_pipeline.py` – Ballbesitz-Berechnung
  - `analytics_pipeline.py` – CSV-Export und Analyse-Trigger
  - `report_pipeline.py` – PDF-Report aus Notebook erzeugen

- `calibration/`
  - Entzerrung (GoPro)
  - Klickbasierte Homographie (Kamera ↔ Spielfeld)
  - Export der Homographie in Meterkoordinaten

- `analytics/`
  - Heatmaps, Pass-Maps, Formationen, Zonenanalysen, Momentum

- `trackers/`
  - `Tracker`-Klasse (YOLO + ByteTrack, Rollenstabilisierung, Rendering)

- `utils/`
  - BBox-Helfer
  - Homographie-Utilities (`pixel_to_pitch`, `pitch_to_pixel`)

> Die genauen Dateinamen und Pfade werden zentral über `config.py` definiert.

---

## Voraussetzungen

- Python 3.10+
- OpenCV (`cv2`)
- `ultralytics`
- `supervision`
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `scikit-learn`
- `mplsoccer`
- `tqdm`
- Jupyter + `nbconvert`
- Google Chrome (für PDF-Export unter macOS)

---

## Installation

### 1. Virtuelle Umgebung erstellen
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 3. Konfiguration prüfen

- `config.py` öffnen und sicherstellen, dass folgende Pfade korrekt gesetzt sind:
  - Input-Video
  - Pitch-Bild
  - YOLO-Modell
  - Kalibrierungsdateien
  - Output-Verzeichnisse (Video, CSV, Plots, Report)

---

## Kalibrierung

Die Pipeline unterstützt zwei Modi:
- **Pixel-Modus**: funktioniert ohne Homographie, eingeschränkte Analysen
- **World-Modus (Meter)**: empfohlen, benötigt Homographie

### Schritt 1: Linsen-Entzerrung (GoPro)
- Nutzung einer gespeicherten Kalibrierungsdatei (`calib_file`)
- Falls nicht vorhanden oder ungültig:
  - Manuelle Justierung über Slider (`k1`, `k2`)

### Schritt 2: Homographie (Klickpunkte)
- Entzerrtes Kamerabild und Pitch-Map öffnen
- Mindestens 4 korrespondierende Punkte klicken
- Ergebnis: Pixel-Homographie (`homography_npy`)

### Schritt 3: Export in Meter
- Umrechnung von Pitch-Pixeln in Meter anhand:
  - Pitch-Bildgrösse
  - Reale Feldmasse
- Ergebnis:
  - `homography.npz` mit:
    - `H`: Kamera → Meter
    - `H_inv`: Meter → Kamera

---

## Ausführen der Match-Analyse

Typischer Ablauf der Pipeline:

1. **Tracking**
   - YOLO-Detektion + ByteTrack
   - Optionales Laden/Speichern eines Tracking-Stubs (`.pkl`)
   - Unterstützung von `frame_skip` für Performance

2. **Team-Zuordnung**
   - Bestimmung der Teamfarben aus frühen Frames
   - Teamzuweisung pro Frame
   - Torhüter-Zuordnung über mittlere X-Position

3. **Ballbesitz**
   - Ball → Spieler über minimale Distanz (Meter)
   - Hysterese gegen Flackern
   - Behandlung kurzer Ballflugphasen

4. **Rendering**
   - Direktes Schreiben eines annotierten Videos
   - Overlay mit Uhr, Spielstand, Ballbesitzbalken
   - Optionale Mini-Maps (Positionen + Voronoi)

5. **Analytics-Export**
   - Frame-basiertes CSV als Grundlage aller Auswertungen
   - Generierung von Plots (Pass-Maps, Heatmaps, etc.)

6. **Report**
   - Notebook ausführen
   - HTML → PDF via Chrome Headless

---

## Outputs

Über `settings.paths` konfigurierbare Ausgaben:
- Annotiertes Video
- Frame-CSV (`frame_events_csv`)
- Plots
- Pass-Maps (Team 1 / Team 2)
- Heatmaps
- Zonenanalyse
- Shot Map
- Momentum-Grafik
- PDF-Report

---

## Frame-CSV (Zentrale Datenschnittstelle)

Wichtige Spalten:
- `frame`, `time_sec`
- `ball_visible`
- `ball_x`, `ball_y` (Pixel)
- `ball_x_m`, `ball_y_m` (Meter)
- `ball_speed_m_s`
- `ball_owner_id`, `ball_owner_team`
- `team_ball_control`
- `team1_players_on_pitch`, `team2_players_on_pitch`

Diese CSV ist die Grundlage für alle Analytics-Module.

---

## Wichtige Konfigurationsparameter

Aus `Settings()`:

- **Tracking**
  - `fps`, `frame_skip`
  - `read_tracks_from_stub`
  - `max_ball_interpolation_gap`

- **Pitch**
  - `pitch_length`, `pitch_width`
  - `pitch_margin`
  - `goal_width`, `goal_depth`

- **Ballbesitz**
  - `min_switch_frames`

- **Pässe**
  - `pass_speed_threshold`
  - `pass_min_distance`
  - `pass_min_frames`

- **Schüsse**
  - `shot_speed_threshold`
