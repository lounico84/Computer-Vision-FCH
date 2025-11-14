import numpy as np
import pandas as pd

from utils import get_center_of_bbox, pixel_to_pitch, is_homography_available

# Export a csv where each row represents one video frame
def export_frame_csv1(tracks, team_ball_control, fps, output_path):

    num_frames = len(tracks["players"])
    rows = []

    use_world = is_homography_available()
    if use_world:
        print("[csv_exporter] Homographie verfügbar – Ballkoordinaten in Meter werden exportiert.")
    else:
        print("[csv_exporter] Keine Homographie – Ballkoordinaten bleiben nur in Pixeln.")

    # Für Geschwindigkeitsberechnung
    last_ball_x_m = np.nan
    last_ball_y_m = np.nan
    last_time_sec = np.nan

    for frame_idx in range(num_frames):
        players = tracks["players"][frame_idx]
        gks     = tracks["goalkeepers"][frame_idx]
        ball    = tracks["ball"][frame_idx]

        # Convert frame indey into time in seconds
        time_sec = frame_idx / float(fps)

        # Check if ball is detected in the current frame
        if 1 in ball:
            ball_visible = 1
            ball_bbox = ball[1]["bbox"]
            # Pixel-Zentrum des Balls
            ball_x, ball_y = get_center_of_bbox(ball_bbox)
        else:
            ball_visible = 0
            ball_x, ball_y = np.nan, np.nan

        # Weltkoordinaten in Metern
        if ball_visible and use_world:
            ball_x_m, ball_y_m = pixel_to_pitch(ball_x, ball_y)
        else:
            ball_x_m, ball_y_m = np.nan, np.nan

        # Ball-Geschwindigkeit in m/s (auf Basis der Meter-Koordinaten)
        if (
            ball_visible
            and use_world
            and not np.isnan(last_ball_x_m)
            and not np.isnan(last_ball_y_m)
        ):
            dt = time_sec - last_time_sec
            if dt > 0:
                dx = ball_x_m - last_ball_x_m
                dy = ball_y_m - last_ball_y_m
                ball_speed_m_s = (dx * dx + dy * dy) ** 0.5 / dt
            else:
                ball_speed_m_s = np.nan
        else:
            ball_speed_m_s = np.nan

        # Cache für nächsten Frame aktualisieren
        if ball_visible and use_world:
            last_ball_x_m = ball_x_m
            last_ball_y_m = ball_y_m
            last_time_sec = time_sec

        # Default values if no owner is found
        owner_id = -1
        owner_role = "none"
        owner_team = 0

        # First check if any field player has the ball
        for pid, pdata in players.items():
            if pdata.get("has_ball", False):
                owner_id = pid
                owner_role = "player"
                owner_team = pdata.get("team", 0)
                break

        # If no player owns the ball, check goalkeepers
        if owner_id == -1:
            for gid, gdata in gks.items():
                if gdata.get("has_ball", False):
                    owner_id = gid
                    owner_role = "goalkeeper"
                    owner_team = gdata.get("team", 0)
                    break

        # Use smoothed ball control value for this frame
        if frame_idx < len(team_ball_control):
            team_control = int(team_ball_control[frame_idx])
        else:
            team_control = 0 # safety fallback

        # Count how many players per team are visible in this frame
        team1_players = sum(1 for p in players.values() if p.get("team") == 1)
        team2_players = sum(1 for p in players.values() if p.get("team") == 2)

        # Collect all values into a structured row
        rows.append(
            {
                "frame": frame_idx,
                "time_sec": time_sec,
                "ball_visible": ball_visible,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_x_m": ball_x_m,
                "ball_y_m": ball_y_m,           
                "ball_speed_m_s": ball_speed_m_s,
                "ball_owner_id": owner_id,
                "ball_owner_role": owner_role,
                "ball_owner_team": owner_team,
                "team_ball_control": team_control,
                "team1_players_on_pitch": team1_players,
                "team2_players_on_pitch": team2_players,
            }
        )

    # Convert list of dicts to a dataframe and save to disk
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved data to: {output_path}")