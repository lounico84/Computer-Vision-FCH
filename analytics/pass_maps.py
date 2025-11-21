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
    speed_threshold=5.0,      # minimale Ballgeschwindigkeit, um als "Pass" zu zählen
    min_distance=7.0,         # minimale Passdistanz in Metern
    min_flight_frames=3,      # (ungenutzt, nur für Kompatibilität)
    landing_window_frames=10, # (ungenutzt)
    min_possession_frames=3,  # Spieler muss mind. so viele Frames Ballkontrolle haben
    max_gap_seconds=6.0,      # max. Zeit zwischen Abspiel und neuer Kontrolle
):
    """
    Konservative Pass-Erkennung basierend auf stabilen Ballbesitz-Segmenten.

    Vorgehen:
      - Baue Segmente (owner_id, team, start_idx, end_idx) mit konstantem ball_owner_id.
      - Ein Pass ist ein Übergang von Segment A -> Segment B, wenn:
          * A: echter Spieler (owner != -1, team in {1,2})
          * Länge(A) >= min_possession_frames
          * B: echter Spieler, owner_B != owner_A, team_B in {1,2}
          * Zeitabstand <= max_gap_seconds
          * Zwischen A-Ende und B-Start existiert mind. ein Frame mit
            ball_speed_m_s >= speed_threshold
          * Distanz(Ballposition Abspiel -> Ballposition Annahme) >= min_distance
      - completed = True, wenn Team(A) == Team(B), sonst Fehlpass/Interception.

    Rückgabe: Liste von dicts mit
      {
        "team": Team des Passgebers (1 oder 2),
        "start_x": x in m (Ballposition beim Abspiel),
        "start_y": y in m,
        "end_x": x in m (Ballposition beim ersten Kontakt des Empfängers),
        "end_y": y in m,
        "completed": True/False
      }
    """

    owners = df["ball_owner_id"].to_numpy()
    teams = df["ball_owner_team"].to_numpy()
    ball_x = df["ball_x_m"].to_numpy()
    ball_y = df["ball_y_m"].to_numpy()
    speed = df["ball_speed_m_s"].to_numpy()

    n = len(df)
    passes = []

    if n == 0:
        print("[pass_maps] detect_passes: 0 Pässe erkannt (leerer DF).")
        return passes

    # --- 1) Segmente von konstantem Besitzer bauen ---
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

    # Referenz-FPS für framebasierte Schwellwerte (30 fps als Standard)
    fps_ref = 30.0
    # effektive Mindestanzahl Frames für Ballbesitz, skaliert mit den tatsächlichen fps
    min_possession_frames_eff = max(1, int(round(min_possession_frames * fps / fps_ref)))

    max_gap_frames = int(max_gap_seconds * fps)

    def valid_before(idx):
        """Letzter Frame <= idx mit gültiger Ballposition."""
        for i in range(idx, -1, -1):
            if math.isfinite(ball_x[i]) and math.isfinite(ball_y[i]):
                return i
        return None

    def valid_after(idx):
        """Erster Frame >= idx mit gültiger Ballposition."""
        for i in range(idx, n):
            if math.isfinite(ball_x[i]) and math.isfinite(ball_y[i]):
                return i
        return None

    # --- 2) Von jedem Ballbesitz-Segment A zum nächsten sinnvollen Segment B ---
    for si, (own, team, start_i, end_i) in enumerate(segments):
        # Nur echte Spielerbesitze berücksichtigen
        if own == -1 or team not in (1, 2):
            continue

        seg_len = end_i - start_i + 1
        if seg_len < min_possession_frames_eff:
            # zu kurze Ballkontrolle -> z.B. Zweikampf-Pingpong
            continue

        # Nächstes Segment B suchen
        candidate_j = None
        for sj in range(si + 1, len(segments)):
            own2, team2, start_j, end_j = segments[sj]

            # zeitliche Lücke zu groß -> keine Verbindung mehr
            if start_j - end_i > max_gap_frames:
                break

            # Segmente ohne Besitzer oder ohne Team ignorieren
            if own2 == -1 or team2 not in (1, 2):
                continue

            # Self-Pass (A -> A) ignorieren
            if own2 == own:
                continue

            candidate_j = sj
            break

        if candidate_j is None:
            continue

        own2, team2, start_j, end_j = segments[candidate_j]

        # --- Geschwindigkeitskriterium: es muss einen "Kick" dazwischen geben ---
        # --- Geschwindigkeitskriterium: es muss einen "Kick" dazwischen geben ---
        t0, t1 = end_i, start_j
        if t1 <= t0:
            continue

        # fps-stabiles Zeitfenster für Kick-Detektion (z. B. 0.20 Sekunden)
        kick_window_seconds = 0.20
        kick_window_frames = int(round(kick_window_seconds * fps))

        # Bereich begrenzen: nur t0 .. t0+kick_window_frames prüfen (oder bis t1)
        t_end = min(t1, t0 + kick_window_frames)

        window_speeds = speed[t0 : t_end + 1]
        finite_mask = np.isfinite(window_speeds)
        if not finite_mask.any():
            continue

        # Kick dann vorhanden, wenn mind. ein Frame über Schwellwert liegt
        if not (window_speeds[finite_mask] >= speed_threshold).any():
            continue

        # --- Start-/Endposition anhand der Ballposition bestimmen ---
        start_idx = valid_before(end_i)
        end_idx = valid_after(start_j)

        if start_idx is None or end_idx is None:
            continue

        sx, sy = ball_x[start_idx], ball_y[start_idx]
        ex, ey = ball_x[end_idx], ball_y[end_idx]

        if not (
            math.isfinite(sx)
            and math.isfinite(sy)
            and math.isfinite(ex)
            and math.isfinite(ey)
        ):
            continue

        dist = math.hypot(ex - sx, ey - sy)
        if dist < min_distance:
            # zu kurzer Weg -> eher kurzer Kontakt/Dribbling
            continue

        completed = (team2 == team)

        passes.append(
            {
                "team": int(team),   # Team des Passgebers
                "start_x": float(sx),
                "start_y": float(sy),
                "end_x": float(ex),
                "end_y": float(ey),
                "completed": bool(completed),
            }
        )

    print(f"[pass_maps] detect_passes: {len(passes)} Pässe erkannt.")
    return passes

def _plot_pass_map(
    passes,
    team,
    pitch_length,
    pitch_width,
    out_path,
    pitch_image_path=None,
):
    """
    Zeichnet eine Pass-Map für ein bestimmtes Team:
    - Hintergrund: Spielfeldbild (optional)
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Pass-Map für Team {team} gespeichert unter: {out_path}")


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
        min_flight_frames=analytics_cfg.pass_min_frames
    )

    _plot_pass_map(
        passes,
        team=1,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team1,
        pitch_image_path=pitch_image_path,
    )
    _plot_pass_map(
        passes,
        team=2,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team2,
        pitch_image_path=pitch_image_path,
    )