import numpy as np
import pandas as pd

from utils import get_center_of_bbox


def export_frame_csv1(tracks, team_ball_control, fps, output_path):
    """
    Exportiert pro Frame eine Zeile mit:
    - Frameindex & Zeit
    - Ball-Position & Sichtbarkeit
    - Ballbesitzer (ID, Rolle, Team)
    - geglättetem Team-Ballbesitz
    - Anzahl sichtbarer Spieler pro Team
    """
    num_frames = len(tracks["players"])
    rows = []

    for frame_idx in range(num_frames):
        players = tracks["players"][frame_idx]
        gks     = tracks["goalkeepers"][frame_idx]
        ball    = tracks["ball"][frame_idx]

        # Zeit
        time_sec = frame_idx / float(fps)

        # Ball sichtbar + Position
        if 1 in ball:
            ball_visible = 1
            ball_bbox = ball[1]["bbox"]
            ball_x, ball_y = get_center_of_bbox(ball_bbox)
        else:
            ball_visible = 0
            ball_x, ball_y = np.nan, np.nan

        # Besitzer (Player oder Keeper) finden
        owner_id = -1
        owner_role = "none"
        owner_team = 0

        # Spieler zuerst
        for pid, pdata in players.items():
            if pdata.get("has_ball", False):
                owner_id = pid
                owner_role = "player"
                owner_team = pdata.get("team", 0)
                break

        # Wenn kein Spieler, dann Keeper checken
        if owner_id == -1:
            for gid, gdata in gks.items():
                if gdata.get("has_ball", False):
                    owner_id = gid
                    owner_role = "goalkeeper"
                    owner_team = gdata.get("team", 0)
                    break

        # geglätteter Team-Ballbesitz
        if frame_idx < len(team_ball_control):
            team_control = int(team_ball_control[frame_idx])
        else:
            team_control = 0

        # Anzahl Spieler pro Team im Frame
        team1_players = sum(1 for p in players.values() if p.get("team") == 1)
        team2_players = sum(1 for p in players.values() if p.get("team") == 2)

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

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[CSV Export] Saved frame-level data to: {output_path}")