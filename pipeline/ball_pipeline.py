import numpy as np

from config import Settings
from player_ball_assigner import PlayerBallAssigner


def compute_team_ball_control(tracks, settings: Settings):
    """Berechnet geglättete Team-Ballkontrolle über alle Frames."""
    player_assigner = PlayerBallAssigner()
    ball_cfg = settings.ball_control

    num_frames = len(tracks["players"])
    team_ball_control = []

    last_team = 0
    candidate_team = None
    candidate_count = 0

    for frame_idx in range(num_frames):
        players = tracks["players"][frame_idx]
        goalkeepers = tracks["goalkeepers"][frame_idx]
        ball_dict = tracks["ball"][frame_idx]

        # 1) Ball sichtbar?
        if 1 not in ball_dict:
            team_ball_control.append(0)
            continue

        ball_bbox = ball_dict[1]["bbox"]

        # 2) Spieler & Keeper zusammen
        all_actors = {}
        all_actors.update(players)
        all_actors.update(goalkeepers)

        assigned_id = player_assigner.assign_ball_to_player(all_actors, ball_bbox)

        if assigned_id == -1:
            team_ball_control.append(0)
            continue

        # 3) Team bestimmen + has_ball setzen
        if assigned_id in players:
            players[assigned_id]["has_ball"] = True
            raw_team = players[assigned_id].get("team", 0)
        else:
            goalkeepers[assigned_id]["has_ball"] = True
            raw_team = goalkeepers[assigned_id].get("team", 0)

        if raw_team not in (1, 2):
            team_ball_control.append(0)
            continue

        # 4) Hysterese für Wechsel
        if last_team == 0:
            last_team = raw_team
            candidate_team = None
            candidate_count = 0
        elif raw_team == last_team:
            candidate_team = None
            candidate_count = 0
        else:
            if candidate_team == raw_team:
                candidate_count += 1
            else:
                candidate_team = raw_team
                candidate_count = 1

            if candidate_count >= ball_cfg.min_switch_frames:
                last_team = raw_team
                candidate_team = None
                candidate_count = 0

        team_ball_control.append(last_team)

    return np.array(team_ball_control)