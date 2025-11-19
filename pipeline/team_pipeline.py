import numpy as np
import cv2

from config import Settings
from utils import get_center_of_bbox
from team_assigner import TeamAssigner

# pipeline/team_pipeline.py

import numpy as np
import cv2

from config import Settings
from utils import get_center_of_bbox
from team_assigner import TeamAssigner

# Assign players and referees to teams based on color information and votes
def assign_teams(tracks, settings: Settings):
    """
    Vereinfachte Team-Pipeline:
    - Teamfarben einmal aus den ersten Frames bestimmen
    - In jedem Frame Team basierend auf der aktuellen BBox-Farbe setzen
    - Keine Votes mehr, keine track_id-basierte Logik
    """
    team_assigner = TeamAssigner()

    video_path = str(settings.paths.input_video)
    num_frames = len(tracks["players"])

    # -----------------------------
    # 1) Erste N Frames laden für die Farb-Bestimmung
    # -----------------------------
    sample_frames_team = 10
    sample_frames_ref  = 50
    max_color_frames = min(num_frames, max(sample_frames_team, sample_frames_ref))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sample_frames = []
    for frame_idx in range(max_color_frames):
        ret, frame = cap.read()
        if not ret:
            break
        sample_frames.append(frame)

    # Tracks für die ersten Frames anpassen
    sample_tracks = {
        "players":   [tracks["players"][i]   for i in range(len(sample_frames))],
        "referees":  [tracks["referees"][i]  for i in range(len(sample_frames))],
        "goalkeepers":[tracks["goalkeepers"][i] for i in range(len(sample_frames))],
        "ball":      [tracks["ball"][i]      for i in range(len(sample_frames))],
    }

    cap.release()

    # -----------------------------
    # 2) Team- und Schirifarben bestimmen
    # -----------------------------
    team_assigner.assign_team_color(sample_frames, sample_tracks, sample_frames=sample_frames_team)
    team_assigner.assign_referee_color(sample_frames, sample_tracks, sample_frames=sample_frames_ref)
    team_assigner.save_color_debug(str(settings.paths.color_debug_image))

    # -----------------------------
    # 3) Vollständigen Videodurchlauf: Team pro Frame setzen
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Einfacher "sticky" Filter: bei unsicheren Frames altes Team behalten
    last_team: dict[int, int] = {}  # track_id -> zuletzt verwendetes Team

    for f_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        for player_id, track in tracks["players"][f_idx].items():
            bbox = track["bbox"]

            # Farbe des Spielers in diesem Frame
            color = team_assigner.get_player_color(frame, bbox).astype(np.float32)

            # Distanz zu den beiden Teamfarben (BGR)
            c1 = np.asarray(team_assigner.team_colors[1], dtype=np.float32)
            c2 = np.asarray(team_assigner.team_colors[2], dtype=np.float32)
            d1 = np.linalg.norm(color - c1)
            d2 = np.linalg.norm(color - c2)

            # rohe Entscheidung: näher an Team 1 oder 2?
            raw_team = 1 if d1 < d2 else 2

            # Heuristik: unsichere Frames erkennen
            #  - d1 und d2 sehr ähnlich  -> Farbe liegt "zwischen" den Teams
            #  - beide Distanzen sehr groß -> Farbe passt zu keinem Team (Schatten, Rasen, etc.)
            if abs(d1 - d2) < 15 or min(d1, d2) > 40:
                # unsicher -> wenn es einen bisherigen Wert gibt, bleib dabei
                team_id = last_team.get(player_id, raw_team)
            else:
                # klarer Fall -> neue Entscheidung übernehmen
                team_id = raw_team

            track["team"] = team_id
            track["team_color"] = team_assigner.team_colors[team_id]
            last_team[player_id] = team_id


    cap.release()
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