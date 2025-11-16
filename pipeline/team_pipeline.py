import numpy as np
import cv2

from config import Settings
from utils import get_center_of_bbox
from team_assigner import TeamAssigner

# Assign players and referees to teams based on color information and votes
def assign_teams(tracks, settings: Settings):
    team_assigner = TeamAssigner()

    video_path = str(settings.paths.input_video)
    num_frames = len(tracks["players"])

    # 1) Wenige Frames für Farb-Schätzung laden
    #    (gleichmäßig über das Video verteilt)
    # Wie viele Frames maximal für K-Means genutzt werden sollen
    sample_frames_team = 500
    sample_frames_ref  = 500

    num_samples = min(num_frames, max(sample_frames_team, sample_frames_ref))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Gleichmäßig verteilte Frame-Indizes bestimmen
    sample_indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
    sample_index_set = set(sample_indices.tolist())

    sample_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in sample_index_set:
            sample_frames.append(frame)

        frame_idx += 1
        if len(sample_frames) >= num_samples:
            break

    # Build a matching tracks subset so that indices of frames and tracks align
    sample_tracks = {
        "players": [],
        "referees": [],
        "goalkeepers": [],
        "ball": [],
    }
    for idx in sample_indices:
        sample_tracks["players"].append(tracks["players"][idx])
        sample_tracks["referees"].append(tracks["referees"][idx])
        sample_tracks["goalkeepers"].append(tracks["goalkeepers"][idx])
        sample_tracks["ball"].append(tracks["ball"][idx])


    cap.release()

    # Team- und Schiri-Farben aus wenigen Frames schätzen
    team_assigner.assign_team_color(sample_frames, sample_tracks, sample_frames=sample_frames_team)
    team_assigner.assign_referee_color(sample_frames, sample_tracks, sample_frames=sample_frames_ref)
    team_assigner.save_color_debug(str(settings.paths.color_debug_image))

    # -----------------------------
    # 2) Zweiter Durchlauf: alle Frames streamen,
    #    um Team-Votes pro Spieler zu sammeln
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        player_track = tracks["players"][frame_idx]
        for player_id, track in player_track.items():
            team_assigner.get_player_team(frame, track["bbox"], player_id)

        frame_idx += 1

    cap.release()

        # -----------------------------
    # 3) Mehrheitsteam pro Spieler bestimmen (wie früher)
    # -----------------------------
    for player_id, votes_dict in team_assigner.team_vote_counts.items():
        votes_team1 = votes_dict.get(1, 0)
        votes_team2 = votes_dict.get(2, 0)
        # Mehrheits-Team (bei Gleichstand Team 1)
        majority_team = 1 if votes_team1 >= votes_team2 else 2
        team_assigner.player_team_dict[player_id] = majority_team

    # -----------------------------
    # 3) Finale Labels berechnen (wie vorher)
    # -----------------------------
    cfg = settings.referee
    final_labels = {}  # mapping player_id: 0 (referee), 1 (team 1), 2 (team 2)

    for player_id, obs_count in team_assigner.player_obs_counts.items():
        ref_votes = team_assigner.ref_vote_counts.get(player_id, 0)
        default_team = team_assigner.player_team_dict.get(player_id, 1)

        if (
            team_assigner.referee_color is not None
            and obs_count >= cfg.min_observations
            and ref_votes >= cfg.min_votes
            and ref_votes >= cfg.min_ratio * obs_count
        ):
            final_labels[player_id] = 0
        else:
            final_labels[player_id] = default_team

    # Tracks umschreiben (wie vorher)
    for frame_idx, player_track in enumerate(tracks["players"]):
        for player_id, track in list(player_track.items()):
            team = final_labels.get(player_id, 1)

            if team == 0:
                tracks["referees"][frame_idx][player_id] = track
                del tracks["players"][frame_idx][player_id]
            else:
                tracks["players"][frame_idx][player_id]["team"] = team
                tracks["players"][frame_idx][player_id]["team_color"] = (
                    team_assigner.team_colors[team]
                )

    return tracks, team_assigner

# Assign each goalkeeper to a team based on thei average horizontal position
## Compute the average x-position of all outfield players per team
## Compute the average x-position of each goalkeeper
## Assign each goalkeeper to the team whose players are closest in x-position
def assign_goalkeepers_to_teams(tracks, team_assigner: TeamAssigner):
    
    # Collect x-coordinates for palyers of each team
    team_x_positions = {1: [], 2: []}

    for player_track in tracks["players"]:
        for track in player_track.values():
            team = track.get("team")
            if team not in (1, 2):
                continue
            x, _ = get_center_of_bbox(track["bbox"])
            team_x_positions[team].append(x)

    # Compute mean x-position for each team
    team_mean_x = {
        team: float(np.mean(xs)) if xs else None
        for team, xs in team_x_positions.items()
    }

    # Collect x-coordinates for each goalkeeper across frames
    goalkeeper_x_positions = {}

    for gk_track in tracks["goalkeepers"]:
        for gk_id, track in gk_track.items():
            x, _ = get_center_of_bbox(track["bbox"])
            goalkeeper_x_positions.setdefault(gk_id, []).append(x)

    # Compute mean x-position for each goalkeeper
    gk_mean_x = {
        gk_id: float(np.mean(xs)) for gk_id, xs in goalkeeper_x_positions.items()
    }

    # Decide for each goalkeeper wich team they belong to
    goalkeeper_team_map = {}

    for gk_id, gk_x in gk_mean_x.items():
        best_team = None
        best_dist = float("inf")

        # Compare distance to each team's mean x-position
        for team in (1, 2):
            mean_x = team_mean_x[team]
            if mean_x is None:
                continue
            dist = abs(gk_x - mean_x)
            if dist < best_dist:
                best_dist = dist
                best_team = team

        # Fallback to team 1
        if best_team is None:
            best_team = 1

        goalkeeper_team_map[gk_id] = best_team

    # Write assigned team labels back into the goalkeepers tracks
    for frame_idx, gk_track in enumerate(tracks["goalkeepers"]):
        for gk_id, track in gk_track.items():
            team = goalkeeper_team_map.get(gk_id, 1)
            tracks["goalkeepers"][frame_idx][gk_id]["team"] = team

    return tracks