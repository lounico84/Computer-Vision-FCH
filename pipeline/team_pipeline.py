import numpy as np
import cv2
from tqdm import tqdm
import pickle
import os
import math
from typing import Dict, Any

from config import Settings
from utils import get_center_of_bbox
from team_assigner import TeamAssigner


# Assign team labels to players (and infer referee color) using jersey-color distance with a simple stability filter
def assign_teams(tracks,
                 settings: Settings,
                 stub_path: str | None = None,
                 read_from_stub: bool = False,
                 resume_from_stub: bool = False,
                 save_stub: bool = True,
                 frame_skip: int = 1
):

    # Load cached team assignments to speed up repeated runs and optionally resume unfinished processing
    if stub_path and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            cached_tracks, cached_team_assigner = pickle.load(f)

        already_done = len(cached_tracks["players"])
        total_frames = len(tracks["players"])

        # Resume from the last processed frame while preserving previously assigned labels
        if resume_from_stub and already_done < total_frames:
            print(f"[STEP 2] Resume: found {already_done} frames, resuming until {total_frames}...")
            for key in ("players", "referees", "goalkeepers"):
                for i in range(already_done):
                    tracks[key][i] = cached_tracks[key][i]

            team_assigner = cached_team_assigner
            start_frame = already_done

        # Return cached results as-is when a full stub is requested
        elif read_from_stub:
            print(f"[STEP 2] Loaded complete team data from {stub_path}")
            return cached_tracks, cached_team_assigner

        # Fall back to a fresh run when stubs exist but are not used
        else:
            start_frame = 0
            team_assigner = TeamAssigner()
    else:
        start_frame = 0
        team_assigner = TeamAssigner()

    video_path = str(settings.paths.input_video)
    num_frames = len(tracks["players"])

    sample_frames_team = 10
    sample_frames_ref  = 50
    max_color_frames = min(num_frames, max(sample_frames_team, sample_frames_ref))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sample_frames = []
    if frame_skip < 1:
        frame_skip = 1

    # Read a small frame subset using the same skip logic as tracking for alignment
    for frame_idx in range(max_color_frames):
        ret, frame = cap.read()
        if not ret:
            break
        sample_frames.append(frame)

        for _ in range(frame_skip - 1):
            ret_skip, _ = cap.read()
            if not ret_skip:
                break

    # Build a matching subset of tracks to keep color estimation consistent with sample frames
    sample_tracks = {
        "players":   [tracks["players"][i]   for i in range(len(sample_frames))],
        "referees":  [tracks["referees"][i]  for i in range(len(sample_frames))],
        "goalkeepers":[tracks["goalkeepers"][i] for i in range(len(sample_frames))],
        "ball":      [tracks["ball"][i]      for i in range(len(sample_frames))],
    }

    cap.release()

    team_assigner.assign_team_color(sample_frames, sample_tracks, sample_frames=sample_frames_team)
    team_assigner.assign_referee_color(sample_frames, sample_tracks, sample_frames=sample_frames_ref)
    team_assigner.save_color_debug(str(settings.paths.color_debug_image))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    if frame_skip < 1:
        frame_skip = 1

    physical_start_frame = start_frame * frame_skip
    cap.set(cv2.CAP_PROP_POS_FRAMES, physical_start_frame)

    # Retain last confident team per track_id to reduce flicker in ambiguous color conditions
    last_team: dict[int, int] = {}

    c1 = np.asarray(team_assigner.team_colors[1], dtype=np.float32)
    c2 = np.asarray(team_assigner.team_colors[2], dtype=np.float32)

    for f_idx in tqdm(range(start_frame, num_frames), total=num_frames - start_frame, desc="Progress"):
        ret, frame = cap.read()
        if not ret:
            break

        for player_id, track in tracks["players"][f_idx].items():
            bbox = track["bbox"]

            # Extract representative jersey color from the player's bounding box for this frame
            color = team_assigner.get_player_color(frame, bbox).astype(np.float32)

            # Compare distances to learned team color prototypes (BGR space)
            d1 = np.linalg.norm(color - c1)
            d2 = np.linalg.norm(color - c2)

            # Choose the closest prototype as the baseline team label
            raw_team = 1 if d1 < d2 else 2

            # Treat near-ties or large distances as uncertain and fall back to last stable label
            if abs(d1 - d2) < 15 or min(d1, d2) > 40:
                team_id = last_team.get(player_id, raw_team)
            else:
                team_id = raw_team

            # Persist team labels and canonical colors back into the track structure
            track["team"] = team_id
            track["team_color"] = team_assigner.team_colors[team_id]
            last_team[player_id] = team_id

        # Advance the capture to keep video frames aligned with tracked frame indices
        for _ in range(frame_skip - 1):
            ret_skip, _ = cap.read()
            if not ret_skip:
                break

    cap.release()

    # Cache the enriched tracks and trained assigner to accelerate future runs
    if save_stub and stub_path:
        with open(stub_path, "wb") as f:
            pickle.dump((tracks, team_assigner), f)

    return tracks, team_assigner


# Assign each goalkeeper to a team by comparing their average x-position to each team's outfield mean x-position
def assign_goalkeepers_to_teams(tracks, team_assigner: TeamAssigner):
    
    # Collect per-team player center x-positions across frames to estimate team side locations
    team_x_positions = {1: [], 2: []}

    for player_track in tracks["players"]:
        for track in player_track.values():
            team = track.get("team")
            if team not in (1, 2):
                continue
            x, _ = get_center_of_bbox(track["bbox"])
            team_x_positions[team].append(x)

    # Compute mean x-position per team as a stable reference for goalkeeper affiliation
    team_mean_x = {
        team: float(np.mean(xs)) if xs else None
        for team, xs in team_x_positions.items()
    }

    # Aggregate goalkeeper center x-positions over time to stabilize assignment
    goalkeeper_x_positions = {}

    for gk_track in tracks["goalkeepers"]:
        for gk_id, track in gk_track.items():
            x, _ = get_center_of_bbox(track["bbox"])
            goalkeeper_x_positions.setdefault(gk_id, []).append(x)

    # Compute mean x-position per goalkeeper as a summary side indicator
    gk_mean_x = {
        gk_id: float(np.mean(xs)) for gk_id, xs in goalkeeper_x_positions.items()
    }

    # Select the closest team side for each goalkeeper based on mean x-distance
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

        # Default to Team 1 when no team reference is available
        if best_team is None:
            best_team = 1

        goalkeeper_team_map[gk_id] = best_team

    # Write goalkeeper team labels back into the per-frame goalkeeper tracks
    for frame_idx, gk_track in enumerate(tracks["goalkeepers"]):
        for gk_id, track in gk_track.items():
            team = goalkeeper_team_map.get(gk_id, 1)
            tracks["goalkeepers"][frame_idx][gk_id]["team"] = team

    return tracks