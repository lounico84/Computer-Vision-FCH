import numpy as np

from config import Settings
from player_ball_assigner import PlayerBallAssigner


def compute_team_ball_control(tracks, settings: Settings):
    """Compute team ball posession over all frames"""
    player_assigner = PlayerBallAssigner()
    ball_cfg = settings.ball_control

    num_frames = len(tracks["players"])
    team_ball_control = []

    # Variables used for team posession
    last_team = 0           # last confirmed team in control
    candidate_team = None   # potential new team taking over
    candidate_count = 0     # how many consecutive frames this new team appears

    for frame_idx in range(num_frames):
        players = tracks["players"][frame_idx]
        goalkeepers = tracks["goalkeepers"][frame_idx]
        ball_dict = tracks["ball"][frame_idx]

        # Check if the ball is visible in this frame
        if 1 not in ball_dict:
            team_ball_control.append(0)
            continue

        ball_bbox = ball_dict[1]["bbox"]

        # Merge players and goalkeepers into one dictionary
        ## Single assignment check
        all_actors = {}
        all_actors.update(players)
        all_actors.update(goalkeepers)

        # Predict who is closest to the ball
        assigned_id = player_assigner.assign_ball_to_player(all_actors, ball_bbox)

        # No valid owner found
        if assigned_id == -1:
            team_ball_control.append(0)
            continue

        # Determine owner's team and mark "has_ball"
        if assigned_id in players:
            players[assigned_id]["has_ball"] = True
            raw_team = players[assigned_id].get("team", 0)
        else:
            goalkeepers[assigned_id]["has_ball"] = True
            raw_team = goalkeepers[assigned_id].get("team", 0)

        # If team not recognized, skip the frame
        if raw_team not in (1, 2):
            team_ball_control.append(0)
            continue

        # Apply hysteresis to smooth team switching
        ## Avoids rapid changes between teams due to noisy detections
        if last_team == 0:
            # If no team has been confirmed yet, accept the first one
            last_team = raw_team
            candidate_team = None
            candidate_count = 0
        elif raw_team == last_team:
            # Same team, reset candidate
            candidate_team = None
            candidate_count = 0
        else:
            # Different team detected, must confirm stability
            if candidate_team == raw_team:
                candidate_count += 1
            else:
                # New team candidate detected
                candidate_team = raw_team
                candidate_count = 1

            # Accept the new team only if it appears consitently for enough frames
            if candidate_count >= ball_cfg.min_switch_frames:
                last_team = raw_team
                candidate_team = None
                candidate_count = 0

        # Store the team control value
        team_ball_control.append(last_team)

    return np.array(team_ball_control)