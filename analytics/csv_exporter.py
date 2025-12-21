import numpy as np
import pandas as pd
from utils import get_center_of_bbox, pixel_to_pitch, is_homography_available
from config import Settings

# Load global analytics configuration
settings = Settings()
analytics_cfg = settings.analytics

# Export a CSV where each row represents one video frame
def export_frame_csv1(tracks, team_ball_control, fps, output_path):

    # Determine total number of frames based on player tracking
    num_frames = len(tracks["players"])
    rows = []

    # Check whether pixel-to-world transformation is available
    use_world = is_homography_available()
    if use_world:
        print("[csv_exporter] Homographie verfügbar – Ballkoordinaten in Meter werden exportiert.")
    else:
        print("[csv_exporter] Keine Homographie – Ballkoordinaten bleiben nur in Pixeln.")

    # Cache previous ball position and timestamp for speed computation
    last_ball_x_m = np.nan
    last_ball_y_m = np.nan
    last_time_sec = np.nan

    for frame_idx in range(num_frames):
        # Extract tracked objects for the current frame
        players = tracks["players"][frame_idx]
        gks     = tracks["goalkeepers"][frame_idx]
        ball    = tracks["ball"][frame_idx]

        # Convert frame index to timestamp in seconds
        time_sec = frame_idx / float(fps)

        # Determine ball visibility and compute pixel center if detected
        if 1 in ball:
            ball_visible = 1
            ball_bbox = ball[1]["bbox"]
            ball_x, ball_y = get_center_of_bbox(ball_bbox)
        else:
            ball_visible = 0
            ball_x, ball_y = np.nan, np.nan

        # Transform ball position to world coordinates if calibration is available
        if ball_visible and use_world:
            ball_x_m, ball_y_m = pixel_to_pitch(ball_x, ball_y)

            # Invalidate positions clearly outside pitch boundaries
            if (
                ball_x_m < -analytics_cfg.pitch_margin
                or ball_x_m > analytics_cfg.pitch_length + analytics_cfg.pitch_margin
                or ball_y_m < -analytics_cfg.pitch_margin
                or ball_y_m > analytics_cfg.pitch_width + analytics_cfg.pitch_margin
            ):
                ball_x_m, ball_y_m = np.nan, np.nan
                ball_visible = 0
        else:
            ball_x_m, ball_y_m = np.nan, np.nan

        # Compute ball speed in m/s using consecutive world positions
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
                speed = (dx * dx + dy * dy) ** 0.5 / dt

                # Filter out physically implausible speed values
                if speed > analytics_cfg.max_ball_speed:
                    ball_speed_m_s = np.nan
                else:
                    ball_speed_m_s = speed
            else:
                ball_speed_m_s = np.nan
        else:
            ball_speed_m_s = np.nan

        # Update cached values for next-frame speed estimation
        if ball_visible and use_world:
            last_ball_x_m = ball_x_m
            last_ball_y_m = ball_y_m
            last_time_sec = time_sec

        # Initialize default ball ownership metadata
        owner_id = -1
        owner_role = "none"
        owner_team = 0

        # Identify ball owner among field players
        for pid, pdata in players.items():
            if pdata.get("has_ball", False):
                owner_id = pid
                owner_role = "player"
                owner_team = pdata.get("team", 0)
                break

        # Fallback to goalkeeper ownership if no player owns the ball
        if owner_id == -1:
            for gid, gdata in gks.items():
                if gdata.get("has_ball", False):
                    owner_id = gid
                    owner_role = "goalkeeper"
                    owner_team = gdata.get("team", 0)
                    break

        # Retrieve smoothed team ball control value for the frame
        if frame_idx < len(team_ball_control):
            team_control = int(team_ball_control[frame_idx])
        else:
            team_control = 0  # Safety fallback for out-of-range access

        # Count visible players per team in the current frame
        team1_players = sum(1 for p in players.values() if p.get("team") == 1)
        team2_players = sum(1 for p in players.values() if p.get("team") == 2)

        # Aggregate all per-frame metrics into a single record
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

    # Persist the aggregated frame-level data as a CSV file
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved data to: {output_path}")