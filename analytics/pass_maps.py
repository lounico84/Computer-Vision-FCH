import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math

from config import Settings
s = Settings()
analytics_cfg = s.analytics


def detect_passes(
    df,
    fps,
    speed_threshold=4.5,      # geringere Schwelle, da Ballgeschwindigkeit vorher geglättet wird
    min_distance=5.0,         # min. 5m – sonst Dribbling/Abgabe
    min_possession_frames=4,  # Sender muss den Ball stabil gehabt haben
    kick_window_seconds=0.12, # engeres Kick-Fenster -> realistischere Passmomente
    max_gap_seconds=3.0       # Empfänger muss innerhalb dieses Fensters Ball erhalten
):
    """
    Verbesserte, extrem robuste Pass-Erkennung.
    Nutzt stabilisierte Besitzdaten + smoothe Ballbahn (Savitzky-Golay).

    Bedingungen:
      - A: Segment von owner_id X
      - B: Nächstes Segment von owner_id Y, Y != X
      - In t0..t0+kick_window muss ball_speed >= threshold sein
      - Flugdistanz >= min_distance
      - B beginnt nicht mehr als max_gap_seconds nach A
    """

    # -----------------------------------------------------
    # 1) Nützliche Arrays extrahieren
    # -----------------------------------------------------
    owners = df["ball_owner_id"].to_numpy()
    teams = df["ball_owner_team"].to_numpy()
    ball_x = df["ball_x_m"].to_numpy()
    ball_y = df["ball_y_m"].to_numpy()
    speed_raw = df["ball_speed_m_s"].to_numpy()

    n = len(df)
    if n < 5:
        return []

    # -----------------------------------------------------
    # 2) Glättung der Ballbahn (Savitzky-Golay)
    # -----------------------------------------------------
    from scipy.signal import savgol_filter

    window = 7 if n >= 7 else (n // 2 * 2 + 1)
    if window < 5:
        window = 5

    x_smooth = savgol_filter(ball_x, window_length=window, polyorder=2, mode="nearest")
    y_smooth = savgol_filter(ball_y, window_length=window, polyorder=2, mode="nearest")

    # neue Geschwindigkeit berechnen
    speed = np.sqrt(np.diff(x_smooth, prepend=x_smooth[0])**2 +
                    np.diff(y_smooth, prepend=y_smooth[0])**2) * fps

    # -----------------------------------------------------
    # 3) Besitzersegmente bauen (owner_id konstant)
    # -----------------------------------------------------
    segments = []
    cur_owner = owners[0]
    cur_team = teams[0]
    start = 0

    for i in range(1, n):
        if owners[i] != cur_owner:
            segments.append((cur_owner, cur_team, start, i - 1))
            cur_owner = owners[i]
            cur_team = teams[i]
            start = i

    segments.append((cur_owner, cur_team, start, n - 1))

    # Zeitskalierung
    min_len = max(1, int(min_possession_frames * fps / 30))
    max_gap_frames = int(max_gap_seconds * fps)
    kick_frames = int(kick_window_seconds * fps)

    passes = []

    def nearest_valid_before(idx):
        for j in range(idx, -1, -1):
            if np.isfinite(x_smooth[j]) and np.isfinite(y_smooth[j]):
                return j
        return None

    def nearest_valid_after(idx):
        for j in range(idx, n):
            if np.isfinite(x_smooth[j]) and np.isfinite(y_smooth[j]):
                return j
        return None

    # -----------------------------------------------------
    # 4) Segmente A → B analysieren
    # -----------------------------------------------------
    for si, (ownA, teamA, startA, endA) in enumerate(segments):

        if ownA == -1 or teamA not in (1, 2):
            continue

        if (endA - startA + 1) < min_len:
            continue  # Ballbesitz zu kurz → Dribbling / Zweikampf

        # Nächstes sinnvolles B-Segment
        target = None
        for sj in range(si + 1, len(segments)):
            ownB, teamB, startB, endB = segments[sj]

            if startB - endA > max_gap_frames:
                break

            if ownB == -1 or teamB not in (1, 2):
                continue

            if ownB == ownA:
                continue  # self-pass ignoriere

            target = (ownB, teamB, startB, endB)
            break

        if target is None:
            continue

        ownB, teamB, startB, endB = target

        # -------------------------------------------------
        # 5) Kick-Detektion (A-Ende → Anfang B)
        # -------------------------------------------------
        t0 = endA
        t_end = min(endA + kick_frames, startB)

        kick_window = speed[t0:t_end+1]
        if np.nanmax(kick_window) < speed_threshold:
            continue  # kein Kick → kein Pass

        # -------------------------------------------------
        # 6) Start-/Endposition bestimmen
        # -------------------------------------------------
        s_idx = nearest_valid_before(endA)
        e_idx = nearest_valid_after(startB)

        if s_idx is None or e_idx is None:
            continue

        sx, sy = x_smooth[s_idx], y_smooth[s_idx]
        ex, ey = x_smooth[e_idx], y_smooth[e_idx]

        if not np.isfinite(sx) or not np.isfinite(ex):
            continue

        dist = math.hypot(ex - sx, ey - sy)
        if dist < min_distance:
            continue  # Dribbling / kleiner Tap / Zweikampf

        completed = (teamA == teamB)

        passes.append({
            "team": int(teamA),
            "start_x": float(sx),
            "start_y": float(sy),
            "end_x": float(ex),
            "end_y": float(ey),
            "completed": bool(completed),
        })

    print(f"[pass_maps] detect_passes: {len(passes)} Pässe erkannt.")
    return passes

def classify_pass_types(
    passes,
    pitch_length,
    clearance_min_distance=25.0,
    defensive_third_ratio=1.0 / 3.0,
):
    """
    Annotiert jeden Pass in 'passes' mit einem einfachen Typ:
      - "clearance": langer Befreiungsschlag aus dem eigenen Defensivdrittel
      - "completed_pass": angekommener Pass
      - "failed_pass": versuchter Pass, der beim Gegner landet / verloren geht

    Heuristik:
      - Defensivdrittel:
          Team 1: x <= pitch_length * defensive_third_ratio
          Team 2: x >= pitch_length * (1 - defensive_third_ratio)
      - Clearance:
          Start im Defensivdrittel UND Distanz >= clearance_min_distance
    """

    for p in passes:
        team = p.get("team", 0)
        sx = p.get("start_x", 0.0)
        ex = p.get("end_x", 0.0)
        sy = p.get("start_y", 0.0)
        ey = p.get("end_y", 0.0)
        completed = p.get("completed", False)

        dx = ex - sx
        dy = ey - sy
        dist = (dx * dx + dy * dy) ** 0.5

        # Defensivdrittel je nach Team
        if team == 1:
            in_def_third = sx <= pitch_length * defensive_third_ratio
        elif team == 2:
            in_def_third = sx >= pitch_length * (1.0 - defensive_third_ratio)
        else:
            in_def_third = False

        # Default-Typ
        pass_type = "completed_pass" if completed else "failed_pass"

        # Clearance-Heuristik
        if in_def_third and dist >= clearance_min_distance:
            pass_type = "clearance"

        p["type"] = pass_type

    return passes

def plot_pass_map(
    passes,
    team,
    pitch_length,
    pitch_width,
    out_path,
    pitch_image_path=None,
):
    """
    Zeichnet eine Pass-Map für ein bestimmtes Team:
    - Hintergrund: Spielfeldbild
    - blaue Linien = angekommene Pässe
    - rote Linien  = Fehlpässe
    """
    team_passes = [p for p in passes if p["team"] == team]
    completed = [p for p in team_passes if p["completed"]]
    failed = [p for p in team_passes if not p["completed"]]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Achsen auf Feldgrösse setzen
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # y=0 oben, y=pitch_width unten

    # ===== Hintergrundbild zeichnen (falls angegeben) =====
    if pitch_image_path is not None:
        img = cv2.imread(pitch_image_path)
        if img is None:
            print(f"[pass_maps] Warnung: Pitch-Image nicht gefunden: {pitch_image_path}")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # extent: [xmin, xmax, ymin, ymax]
            # wegen invert_yaxis() setzen wir ymin=pitch_width, ymax=0
            ax.imshow(
                img,
                extent=[0, pitch_length, pitch_width, 0],
                interpolation="bilinear",
            )
    else:
        # Falls kein Bild: einfachen Rahmen zeichnen
        ax.plot(
            [0, pitch_length, pitch_length, 0, 0],
            [0, 0, pitch_width, pitch_width, 0],
            color="black",
            linewidth=1,
        )

    # ===== Pässe einzeichnen =====

    # Angekommene Pässe: blau
    for p in completed:
        ax.plot(
            [p["start_x"], p["end_x"]],
            [p["start_y"], p["end_y"]],
            color="blue",
            alpha=0.6,
            linewidth=1.0,
        )

    # Fehlpässe: rot
    for p in failed:
        ax.plot(
            [p["start_x"], p["end_x"]],
            [p["start_y"], p["end_y"]],
            color="red",
            alpha=0.6,
            linewidth=1.0,
        )

    ax.set_title(f"Pass Map Team {team}")
    ax.set_xlabel("Länge [m]")
    ax.set_ylabel("Breite [m]")
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        print(f"Pass-Map für Team {team} gespeichert unter: {out_path}")
        plt.close(fig)  # nach dem Speichern schließen
    else:
        # im Notebook anzeigen, nicht schließen
        plt.show()


def create_pass_maps_from_csv(
    csv_path,
    out_path_team1,
    out_path_team2,
    pitch_length=100.0,
    pitch_width=60.0,
    fps=60,
    speed_threshold=5.0,
    min_distance=3.0,
    pitch_image_path=None,
):
    df = pd.read_csv(csv_path)

    required_cols = {
        "ball_x_m",
        "ball_y_m",
        "ball_speed_m_s",
        "team_ball_control",
        "ball_visible",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Folgende Spalten fehlen in {csv_path} und werden für die Pass-Maps benötigt: {missing}"
        )

    passes = detect_passes(
        df,
        fps=fps,
        speed_threshold=speed_threshold,
        min_distance=min_distance,
        min_possession_frames=analytics_cfg.pass_min_frames
    )

    # Pass-Typen klassifizieren (Pass vs Fehlpass vs Klärung)
    passes = classify_pass_types(
        passes,
        pitch_length=pitch_length,
    )

    # Optional: kleine Statistik auf die Konsole
    num_total = len(passes)
    num_clear = sum(1 for p in passes if p.get("type") == "clearance")
    num_completed = sum(1 for p in passes if p.get("type") == "completed_pass")
    num_failed = sum(1 for p in passes if p.get("type") == "failed_pass")
    
    print(
        f"[pass_maps] Summary: total={num_total}, "
        f"completed={num_completed}, failed={num_failed}, clearances={num_clear}"
    )

    # Team 1
    plot_pass_map(
        passes,
        team=1,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team1,
        pitch_image_path=pitch_image_path,
    )

    # Team 2
    plot_pass_map(
        passes,
        team=2,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team2,
        pitch_image_path=pitch_image_path,
    )