# analytics/pass_maps.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_passes(
    df,
    fps,
    speed_threshold=5.0,
    min_distance=1.0,
    min_flight_frames=3,
    landing_window_frames=10,
):
    """
    Erkenne Pässe basierend auf:
      - ball_x_m, ball_y_m
      - ball_speed_m_s
      - team_ball_control

    Logik:
      - Start eines Passes:
          * team_ball_control in {1,2}
          * ball_speed steigt von < threshold auf >= threshold
      - Flugphase:
          * Ball sichtbar, Speed weiterhin hoch
      - Ende:
          * Speed < threshold oder Ball nicht sichtbar
      - Klassifikation:
          * Schaue landing_window_frames Frames nach "Landung"
          * dom. team_ball_control == flight_team -> angekommen
          * dom. anderes Team / kein Team -> Fehlpass
    """

    passes = []

    in_flight = False
    flight_team = None
    start_x = start_y = None
    start_idx = None

    last_speed = np.nan
    last_team_control = 0

    n = len(df)

    for idx, row in df.iterrows():
        visible = row.get("ball_visible", 0)
        x = row.get("ball_x_m", np.nan)
        y = row.get("ball_y_m", np.nan)
        speed = row.get("ball_speed_m_s", np.nan)
        team_ctrl = int(row.get("team_ball_control", 0))

        if not visible or not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(speed):
            last_speed = speed
            last_team_control = team_ctrl
            continue

        # -------- Pass-Start?
        if not in_flight:
            if (
                team_ctrl in (1, 2)
                and (not np.isfinite(last_speed) or last_speed < speed_threshold)
                and speed >= speed_threshold
            ):
                in_flight = True
                flight_team = team_ctrl
                start_x, start_y = x, y
                start_idx = idx

        else:
            # -------- Pass-Flug läuft
            # Ende, wenn Geschwindigkeit wieder klein wird
            if speed < speed_threshold:
                end_x, end_y = x, y
                end_idx = idx
                flight_frames = end_idx - start_idx + 1
                dist = float(np.hypot(end_x - start_x, end_y - start_y))

                if flight_frames >= min_flight_frames and dist >= min_distance:
                    # Landing-Fenster zur Klassifikation
                    look_end = min(end_idx + landing_window_frames, n - 1)
                    post = df.iloc[end_idx:look_end + 1]
                    teams = post["team_ball_control"].to_numpy()

                    # häufigstes Team (ohne 0)
                    valid = teams[(teams == 1) | (teams == 2)]
                    if valid.size > 0:
                        values, counts = np.unique(valid, return_counts=True)
                        new_team = int(values[np.argmax(counts)])
                    else:
                        new_team = 0

                    completed = (new_team == flight_team)

                    passes.append(
                        {
                            "team": flight_team,
                            "start_x": start_x,
                            "start_y": start_y,
                            "end_x": end_x,
                            "end_y": end_y,
                            "completed": completed,
                        }
                    )

                # Reset Flug
                in_flight = False
                flight_team = None
                start_x = start_y = None
                start_idx = None

        last_speed = speed
        last_team_control = team_ctrl

    return passes


def _plot_pass_map(passes, team, pitch_length, pitch_width, out_path):
    """
    Zeichnet eine Pass-Map für ein bestimmtes Team:
    - blaue Linien = angekommene Pässe
    - rote Linien  = Fehlpässe
    """
    team_passes = [p for p in passes if p["team"] == team]
    completed = [p for p in team_passes if p["completed"]]
    failed = [p for p in team_passes if not p["completed"]]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Spielfeldrahmen
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # y=0 oben, y=68 unten (wie im Bild)

    # Randlinien zeichnen
    ax.plot(
        [0, pitch_length, pitch_length, 0, 0],
        [0, 0, pitch_width, pitch_width, 0],
        color="black",
        linewidth=1,
    )

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
    pitch_length=105.0,
    pitch_width=68.0,
    fps=60,
    speed_threshold=5.0,
    min_distance=3.0,
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
    )

    _plot_pass_map(
        passes,
        team=1,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team1,
    )
    _plot_pass_map(
        passes,
        team=2,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        out_path=out_path_team2,
    )