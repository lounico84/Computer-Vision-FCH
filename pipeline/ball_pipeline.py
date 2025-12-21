import numpy as np

from config import Settings
from player_ball_assigner import PlayerBallAssigner


# Compute a stable per-frame team possession signal using owner assignment plus temporal hysteresis
def compute_team_ball_control(tracks, settings: Settings):

    # Initialize the ball-to-player assigner and auto-tune distance thresholds from observed tracks
    player_assigner = PlayerBallAssigner()
    player_assigner.auto_calibrate_from_tracks(tracks, max_frames=600)
    ball_cfg = settings.ball_control

    num_frames = len(tracks["players"])
    team_ball_control = []

    # Maintain team-level hysteresis to avoid flickering possession on close contests
    last_team = 0           # last stable team in control (1/2), 0 = unknown
    candidate_team = None   # potential new team in control
    candidate_count = 0     # consecutive frames supporting the candidate team

    # Maintain owner-level hysteresis to stabilize ball owner identity across noisy assignments
    last_owner_id = None        # last stable owner track_id
    owner_candidate_id = None   # potential new owner track_id
    owner_candidate_count = 0   # consecutive frames supporting the candidate owner

    # Allow short "free ball" periods (e.g., aerial passes) without immediately dropping team control
    max_free_ball_frames = 10
    free_ball_counter = 0

    for frame_idx in range(num_frames):
        players = tracks["players"][frame_idx]
        goalkeepers = tracks["goalkeepers"][frame_idx]
        ball_dict = tracks["ball"][frame_idx]

        # Default ownership for this frame before assignment succeeds
        owner_id = -1
        owner_team = 0

        # Handle missing ball detections by treating the ball as temporarily free
        if 1 not in ball_dict:
            free_ball_counter += 1

            # Keep last known team control during short gaps; otherwise mark possession as unknown
            if last_team != 0 and free_ball_counter <= max_free_ball_frames:
                team_ball_control.append(last_team)
            else:
                team_ball_control.append(0)

            continue

        # Extract current ball bounding box for proximity-based owner assignment
        ball_bbox = ball_dict[1]["bbox"]

        # Build eligible actors (players + goalkeepers) and explicitly exclude referees from ownership
        all_actors = {}
        all_actors.update(players)
        all_actors.update(goalkeepers)

        # Assign ball ownership to the closest eligible actor (distance computed in meters downstream)
        assigned_id = player_assigner.assign_ball_to_player(all_actors, ball_bbox)

        # Handle visible ball without a confident owner as a "free ball" period
        if assigned_id == -1:
            free_ball_counter += 1

            ball_dict[1]["owner_id"] = owner_id
            ball_dict[1]["owner_team"] = owner_team

            # Preserve last team control for short free-ball windows to reduce possession dropouts
            if last_team != 0 and free_ball_counter <= max_free_ball_frames:
                team_ball_control.append(last_team)
            else:
                team_ball_control.append(0)

            continue

        # Reset free-ball counter once an owner candidate is available
        free_ball_counter = 0

        # Stabilize owner identity using hysteresis to avoid rapid switching on borderline distances
        if last_owner_id is None:
            last_owner_id = assigned_id
            owner_candidate_id = None
            owner_candidate_count = 0
        elif assigned_id == last_owner_id:
            owner_candidate_id = None
            owner_candidate_count = 0
        else:
            if owner_candidate_id == assigned_id:
                owner_candidate_count += 1
            else:
                owner_candidate_id = assigned_id
                owner_candidate_count = 1

            # Confirm owner change only after sustained evidence over multiple frames
            if owner_candidate_count >= ball_cfg.min_switch_frames:
                last_owner_id = assigned_id
                owner_candidate_id = None
                owner_candidate_count = 0

        owner_id = last_owner_id

        # Resolve the owner's team and mark has_ball for downstream visualizations and analytics
        raw_team = 0

        if owner_id in players:
            players[owner_id]["has_ball"] = True
            raw_team = players[owner_id].get("team", 0)
        elif owner_id in goalkeepers:
            goalkeepers[owner_id]["has_ball"] = True
            raw_team = goalkeepers[owner_id].get("team", 0)
        else:
            # If the stable owner is not visible, keep team control stable to avoid spurious drops
            ball_dict[1]["owner_id"] = owner_id
            ball_dict[1]["owner_team"] = 0

            team_ball_control.append(last_team)
            continue

        owner_team = raw_team

        # Persist owner metadata on the ball track for CSV export and later event detection
        ball_dict[1]["owner_id"] = owner_id
        ball_dict[1]["owner_team"] = owner_team

        # Fallback to last known team when team assignment is missing/invalid in this frame
        if raw_team not in (1, 2):
            team_ball_control.append(last_team if last_team != 0 else 0)
            continue

        # Stabilize team possession using hysteresis to prevent oscillation on quick challenges
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

            # Confirm team change only after sustained evidence over multiple frames
            if candidate_count >= ball_cfg.min_switch_frames:
                last_team = raw_team
                candidate_team = None
                candidate_count = 0

        team_ball_control.append(last_team)

    # Return possession as a compact NumPy array for downstream smoothing and visualization
    return np.array(team_ball_control)