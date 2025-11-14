import numpy as np
import pandas as pd

from utils import get_center_of_bbox


def export_frame_csv1(tracks, team_ball_control, fps, output_path):
    """Export a csv where each row represents one video frame"""
    num_frames = len(tracks["players"])
    rows = []

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
            # Get the (x,y) center of point of the ball's bounding box
            ball_x, ball_y = get_center_of_bbox(ball_bbox)
        else:
            ball_visible = 0
            ball_x, ball_y = np.nan, np.nan

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