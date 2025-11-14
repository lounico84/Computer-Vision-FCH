# pipeline/team_pipeline.py

import numpy as np

from config import Settings
from utils import get_center_of_bbox
from team_assigner import TeamAssigner

# Assign Player, Ref to Teams
def assign_teams(video_frames, tracks, settings: Settings):
    team_assigner = TeamAssigner()

    # Teamfarben / Referee-Farbe bestimmen
    team_assigner.assign_team_color(video_frames, tracks)
    team_assigner.assign_referee_color(video_frames, tracks)
    team_assigner.save_color_debug(str(settings.paths.color_debug_image))

    # Votes pro Spieler sammeln
    for frame_idx, player_track in enumerate(tracks["players"]):
        frame = video_frames[frame_idx]
        for player_id, track in player_track.items():
            team_assigner.get_player_team(frame, track["bbox"], player_id)

    # Mehrheits-Team pro Spieler
    for player_id, votes_dict in team_assigner.team_vote_counts.items():
        votes_team1 = votes_dict.get(1, 0)
        votes_team2 = votes_dict.get(2, 0)
        majority_team = 1 if votes_team1 >= votes_team2 else 2
        team_assigner.player_team_dict[player_id] = majority_team

    # Finale Entscheidung: Referee vs. Team 1/2
    cfg = settings.referee
    final_labels = {}  # player_id -> 0 (Ref), 1, 2

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

    # Tracks nach finalen Labels umbauen
    for frame_idx, player_track in enumerate(tracks["players"]):
        for player_id, track in list(player_track.items()):
            team = final_labels.get(player_id, 1)

            if team == 0:
                # kompletter Track ist Schiri
                tracks["referees"][frame_idx][player_id] = track
                del tracks["players"][frame_idx][player_id]
            else:
                tracks["players"][frame_idx][player_id]["team"] = team
                tracks["players"][frame_idx][player_id]["team_color"] = (
                    team_assigner.team_colors[team]
                )

    return tracks, team_assigner


def assign_goalkeepers_to_teams(tracks, team_assigner: TeamAssigner):
    """Ordnet Torhüter anhand der mittleren x-Position den Teams zu."""
    # Team-Schwerpunkte
    team_x_positions = {1: [], 2: []}

    for player_track in tracks["players"]:
        for track in player_track.values():
            team = track.get("team")
            if team not in (1, 2):
                continue
            x, _ = get_center_of_bbox(track["bbox"])
            team_x_positions[team].append(x)

    team_mean_x = {
        team: float(np.mean(xs)) if xs else None
        for team, xs in team_x_positions.items()
    }

    # Torhüter-Schwerpunkte
    goalkeeper_x_positions = {}

    for gk_track in tracks["goalkeepers"]:
        for gk_id, track in gk_track.items():
            x, _ = get_center_of_bbox(track["bbox"])
            goalkeeper_x_positions.setdefault(gk_id, []).append(x)

    gk_mean_x = {
        gk_id: float(np.mean(xs)) for gk_id, xs in goalkeeper_x_positions.items()
    }

    # Zuordnung Keeper -> Team
    goalkeeper_team_map = {}

    for gk_id, gk_x in gk_mean_x.items():
        best_team = None
        best_dist = float("inf")

        for team in (1, 2):
            mean_x = team_mean_x[team]
            if mean_x is None:
                continue
            dist = abs(gk_x - mean_x)
            if dist < best_dist:
                best_dist = dist
                best_team = team

        if best_team is None:
            best_team = 1

        goalkeeper_team_map[gk_id] = best_team

    # Teams in den Tracks eintragen
    for frame_idx, gk_track in enumerate(tracks["goalkeepers"]):
        for gk_id, track in gk_track.items():
            team = goalkeeper_team_map.get(gk_id, 1)
            tracks["goalkeepers"][frame_idx][gk_id]["team"] = team
            # Farbe bleibt blau -> kein team_color setzen

    return tracks