import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
from mplsoccer import Pitch

from config import Settings
s = Settings()
analytics_cfg = s.analytics


def detect_passes(
    df,
    fps,
    speed_threshold=4.5,
    min_distance=5.0,
    min_possession_frames=4,
    kick_window_seconds=0.12,
    max_gap_seconds=3.0
):
    """
    Erkennt Pässe basierend auf Ballbesitzwechseln und Ballgeschwindigkeit.
    """
    owners = df["ball_owner_id"].to_numpy()
    teams = df["ball_owner_team"].to_numpy()
    ball_x = df["ball_x_m"].to_numpy()
    ball_y = df["ball_y_m"].to_numpy()

    n = len(df)
    if n < 5:
        return []

    # Glättung der Ballbahn
    from scipy.signal import savgol_filter
    window = 7 if n >= 7 else (n // 2 * 2 + 1)
    if window < 5: window = 5

    x_smooth = savgol_filter(ball_x, window_length=window, polyorder=2, mode="nearest")
    y_smooth = savgol_filter(ball_y, window_length=window, polyorder=2, mode="nearest")

    # Geschwindigkeit berechnen
    speed = np.sqrt(np.diff(x_smooth, prepend=x_smooth[0])**2 +
                    np.diff(y_smooth, prepend=y_smooth[0])**2) * fps

    # Segmente bauen
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

    # Parameter in Frames
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

    # Analyse
    for si, (ownA, teamA, startA, endA) in enumerate(segments):
        if ownA == -1 or teamA not in (1, 2): continue
        if (endA - startA + 1) < min_len: continue

        target = None
        for sj in range(si + 1, len(segments)):
            ownB, teamB, startB, endB = segments[sj]
            if startB - endA > max_gap_frames: break
            if ownB == -1 or teamB not in (1, 2): continue
            if ownB == ownA: continue 
            target = (ownB, teamB, startB, endB)
            break

        if target is None: continue

        ownB, teamB, startB, endB = target

        # Kick-Check
        t0 = endA
        t_end = min(endA + kick_frames, startB)
        kick_window = speed[t0:t_end+1]

        condition = ~np.isnan(kick_window)

        kick_window = kick_window[condition]

        if np.all(np.isnan(kick_window)):
            continue
        
        if len(kick_window) == 0 or np.all(np.isnan(kick_window)):
            continue
        
        if np.nanmax(kick_window) < speed_threshold:
            continue

        # Positionen
        s_idx = nearest_valid_before(endA)
        e_idx = nearest_valid_after(startB)

        if s_idx is None or e_idx is None: continue

        sx, sy = x_smooth[s_idx], y_smooth[s_idx]
        ex, ey = x_smooth[e_idx], y_smooth[e_idx]

        if not np.isfinite(sx) or not np.isfinite(ex): continue

        dist = math.hypot(ex - sx, ey - sy)
        if dist < min_distance: continue

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
    for p in passes:
        team = p.get("team", 0)
        sx = p.get("start_x", 0.0)
        dist = math.hypot(p["end_x"] - sx, p["end_y"] - p["start_y"])
        completed = p.get("completed", False)

        if team == 1:
            in_def_third = sx <= pitch_length * defensive_third_ratio
        elif team == 2:
            in_def_third = sx >= pitch_length * (1.0 - defensive_third_ratio)
        else:
            in_def_third = False

        pass_type = "completed_pass" if completed else "failed_pass"
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
    team_name=None # Neuer Parameter für direkten Namen
):
    """
    Zeichnet eine professionelle Pass-Map im 'Dark Mode' (TV-Style) mit mplsoccer.
    """
    team_cfg = getattr(s, "team_names", None)
    
    # 1. Namen & Farben bestimmen
    if team == 1:
        # Prio 1: Übergebener Name -> Prio 2: Config -> Prio 3: Default
        display_name = team_name if team_name else (team_cfg.team1_name if team_cfg else "Team 1")
        team_color = '#00bfff' # Electric Blue
    else:
        display_name = team_name if team_name else (team_cfg.team2_name if team_cfg else "Team 2")
        team_color = '#dc143c' # Neon Red

    # Daten filtern
    team_passes = [p for p in passes if p["team"] == team]
    completed = [p for p in team_passes if p["completed"]]
    failed = [p for p in team_passes if not p["completed"]]

    # 2. SETUP: Pitch im Dark-Mode
    # pitch_type='custom' erlaubt eigene Maße (Meter)
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        line_color='#c7d5cc',   # Helles Grau für Linien
        pitch_color='#22312b',  # Dunkles Grün (Hintergrund)
        linewidth=2,
    )

    # Figure erstellen
    fig, ax = pitch.draw(figsize=(10, 6))
    
    # WICHTIG: Hintergrund der gesamten Grafik dunkel machen
    # (Sonst ist der weiße Titel auf weißem Rand unsichtbar)
    fig.set_facecolor('#22312b')

    # WICHTIG: Achse drehen (0=Oben)
    ax.invert_yaxis()

    # 3. Pässe zeichnen
    # A) Erfolgreiche Pässe
    if completed:
        pitch.arrows(
            [p["start_x"] for p in completed], [p["start_y"] for p in completed],
            [p["end_x"] for p in completed], [p["end_y"] for p in completed],
            width=2, headwidth=3, headlength=3, 
            color=team_color, ax=ax, label="Angekommen", zorder=2
        )
        # Startpunkte als Nodes
        pitch.scatter(
            [p["start_x"] for p in completed], [p["start_y"] for p in completed],
            s=45, color=team_color, edgecolors='white', linewidth=1, ax=ax, alpha=0.9, zorder=3
        )

    # B) Fehlpässe
    if failed:
        pitch.lines(
            [p["start_x"] for p in failed], [p["start_y"] for p in failed],
            [p["end_x"] for p in failed], [p["end_y"] for p in failed],
            color='#ba4f45', alpha=0.6, lw=1.5, ls='--', ax=ax, label="Fehlpass", zorder=1
        )
        pitch.scatter(
            [p["end_x"] for p in failed], [p["end_y"] for p in failed],
            marker='x', s=40, color='#ba4f45', ax=ax, zorder=2
        )

    # 4. Titel setzen (mit fig.text, damit er sicher oben steht)
    # y=0.96 ist knapp unter dem oberen Rand
    fig.text(0.5, 0.96, f"{display_name} - Passnetzwerk", 
             color='white', fontsize=20, fontweight='bold', 
             ha='center', va='center')
    
    # Legende
    legend = ax.legend(facecolor='#22312b', edgecolor='None', fontsize=10, loc='lower left')
    for text in legend.get_texts():
        text.set_color("white")

    # Speichern oder Anzeigen
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#22312b')
        print(f"Pass-Map für {display_name} gespeichert unter: {out_path}")
        plt.close(fig)
    else:
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

    # Spalten prüfen [cite: 72]
    required_cols = {"ball_x_m", "ball_y_m", "ball_speed_m_s", "team_ball_control", "ball_visible"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Fehlende Spalten in CSV: {missing}")

    passes = detect_passes(
        df,
        fps=fps,
        speed_threshold=speed_threshold,
        min_distance=min_distance,
        min_possession_frames=analytics_cfg.pass_min_frames
    )

    passes = classify_pass_types(passes, pitch_length=pitch_length)

    # Stats
    num_total = len(passes)
    num_completed = sum(1 for p in passes if p.get("type") == "completed_pass")
    num_failed = sum(1 for p in passes if p.get("type") == "failed_pass")
    
    print(f"[pass_maps] Summary: total={num_total}, completed={num_completed}, failed={num_failed}")

    # Plot Team 1
    plot_pass_map(
        passes,
        team=1,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team1,
        pitch_image_path=pitch_image_path,
    )

    # Plot Team 2
    plot_pass_map(
        passes,
        team=2,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team2,
        pitch_image_path=pitch_image_path,
    )