from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from math import ceil
from scipy.spatial import Voronoi

from utils import get_bbox_width, get_center_of_bbox, get_bbox_height, pixel_to_pitch, is_homography_available
from config import Settings

settings = Settings()
analytics_cfg = settings.analytics
team_cfg = settings.team_names


class Tracker:
    # Orchestrate YOLO detection, ByteTrack association, post-processing, and video overlay rendering
    def __init__(self, model_path):
        # Initialize the YOLO model once to reuse weights across all batches
        #device = torch.device("mps")
        #self.model = YOLO(model_path).to(device) # load YOLO model from the given path

        # Use CPU inference for portability when GPU/MPS is not available
        self.model = YOLO(model_path) # load YOLO model from the given path
        self.tracker = sv.ByteTrack() # initialize ByteTrack tracker from supervision
    
    def interpolate_ball_positions(self, ball_positions, max_gap=20, max_jump_px=80):
        # Build a frame-aligned bbox series with NaNs for missing ball detections
        raw = []
        for x in ball_positions:
            if 1 in x:
                raw.append(x[1]['bbox'])
            else:
                raw.append([np.nan, np.nan, np.nan, np.nan])

        # Use a tabular representation to leverage pandas interpolation utilities
        df = pd.DataFrame(raw, columns=['x1','y1','x2','y2'])

        # Convert bbox corners to center points to measure motion continuity
        cx = (df["x1"] + df["x2"]) / 2.0
        cy = (df["y1"] + df["y2"]) / 2.0

        # Flag discontinuities as invalid to prevent interpolation across tracking glitches
        prev_cx = cx.shift(1)
        prev_cy = cy.shift(1)
        dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

        # Drop frames with unrealistic center displacement before interpolation
        mask_big_jump = dist > max_jump_px
        df.loc[mask_big_jump, ["x1","y1","x2","y2"]] = np.nan

        # Interpolate only short gaps to avoid fabricating long, uncertain trajectories
        df = df.interpolate(limit=max_gap, limit_direction="both")

        # Reconstruct per-frame dict format expected by downstream pipeline components
        ball_tracks = []
        for bbox in df.to_numpy().tolist():
            if any(np.isnan(bbox)):
                ball_tracks.append({})
            else:
                ball_tracks.append({1: {"bbox": bbox}})
        return ball_tracks

    def detect_frames(self, frames):
        # Batch inference reduces overhead and improves runtime on CPU/GPU backends
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i+batch_size],
                conf=0.1,
                verbose=False,
            )
            detections += detections_batch
        
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Prefer cached tracks during development to avoid re-running expensive inference
        tracks = None

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)

        else:
            # Run inference and assemble a frame-indexed track dictionary for downstream analytics
            detections = self.detect_frames(frames)

            tracks = {
                "players": [],
                "referees": [],
                "goalkeepers": [],
                "ball": []
            }

            for frame_num, detection in enumerate(detections):
                # Build name->id mapping once per frame to keep class routing robust
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                # Convert YOLO output into a consistent detection container for ByteTrack
                detection_supervision = sv.Detections.from_ultralytics(detection)

                # Enforce stricter ball confidence to reduce false positives on small objects
                ball_class_id = cls_names_inv["ball"]
                ball_conf_min = 0.30

                mask_keep = []
                for cls_id, conf in zip(detection_supervision.class_id, detection_supervision.confidence):
                    mask_keep.append(not (cls_id == ball_class_id and conf < ball_conf_min))
        
                detection_supervision = detection_supervision[mask_keep]

                # Associate detections across frames to obtain stable track IDs
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                # Create empty per-frame containers to keep indexing consistent
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["goalkeepers"].append({})
                tracks["ball"].append({})

                # Route tracked detections into role-specific dictionaries for later team logic
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}

                    if cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    
                    if cls_id == cls_names_inv['goalkeeper']:
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

                # Select a single ball per frame by confidence and physical plausibility on the pitch
                ball_candidates = []

                for frame_detection in detection_supervision:
                    bbox = frame_detection[0]
                    conf = frame_detection[2]
                    cls_id = frame_detection[3]

                    if cls_id != ball_class_id:
                        continue

                    ball_candidates.append((conf, bbox))

                ball_candidates.sort(key=lambda x: x[0], reverse=True)

                for conf, bbox in ball_candidates:
                    x1, y1, x2, y2 = bbox
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)

                    # Validate ball candidates using the homography-based pitch coordinate transform
                    X, Y = pixel_to_pitch(cx, cy)
                    if not np.isfinite(X) or not np.isfinite(Y):
                        continue

                    margin = analytics_cfg.pitch_margin
                    L = analytics_cfg.pitch_length
                    W = analytics_cfg.pitch_width

                    # Keep only candidates inside pitch bounds (plus a small margin for tracking noise)
                    if (
                        X < -margin or X > L + margin or
                        Y < -margin or Y > W + margin
                    ):
                        continue

                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    break
        
        # Stabilize role assignment per track_id to mitigate class flicker across frames
        tracks = self._stabilize_roles_per_track(
            tracks,
            min_observations=5,
            min_ratio=0.6,
        )

        # Persist newly computed tracks for reproducibility and faster iteration
        if stub_path is not None and not read_from_stub:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Anchor the marker at the bbox bottom to approximate foot contact point
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        height = get_bbox_height(bbox)

        # Use an ellipse to reduce occlusion and keep overlays readable at distance
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Scale label box to player size to maintain readability across zoom levels
        rectangle_height = int(max(height * 0.25, 10))
        rectangle_width = int(max(width * 0.70, 25))

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2

        # Place the label just above the marker to avoid covering the entity silhouette
        y1_rect = y2 + 8
        y2_rect = y1_rect + rectangle_height

        if track_id is not None:
            # Draw a filled label background using team color for quick identification
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            # Adjust font size based on box height to keep IDs legible
            font_scale = max(rectangle_height / 22, 0.4)
            text_y = y1_rect + int(rectangle_height * 0.75)

            # Center the ID text to avoid jitter in perception across frames
            text = f"{track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x_center - text_size[0] // 2

            cv2.putText(
                frame,
                text,
                (int(text_x), int(text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        return frame
    
    def draw_triangle(self, frame, bbox, color):
        # Place the marker at bbox top to avoid hiding feet/ball contact cues
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Use a triangle for high salience at small sizes
        traingle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [traingle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traingle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, score1=0, score2=0, event_text=None, fps=None):
        # Default FPS ensures time rendering works even when caller does not pass it explicitly
        if fps is None:
            fps = 30

        h, w = frame.shape[:2]

        # Compute possession ratios from the match-start slice for a stable on-screen KPI
        tbc_slice = team_ball_control[:frame_num + 1]
        if isinstance(tbc_slice, list):
            tbc_slice = np.array(tbc_slice)

        t1_frames = np.sum(tbc_slice == 1)
        t2_frames = np.sum(tbc_slice == 2)
        total = t1_frames + t2_frames

        if total == 0:
            pct1 = pct2 = 0.0
        else:
            pct1 = t1_frames / total
            pct2 = t2_frames / total

        # Pull team colors from current player tracks to keep UI aligned with jersey detection
        color1 = (0, 150, 255)
        color2 = (255, 60, 60)

        players = getattr(self, "current_players", None)
        if players is not None:
            for _, p in players.items():
                raw = p.get("team_color")
                tid = p.get("team")
                if raw is None or tid not in (1, 2):
                    continue
                col = tuple(int(x) for x in np.asarray(raw).tolist())
                if tid == 1:
                    color1 = col
                elif tid == 2:
                    color2 = col

        # Use configured team names and generate 3-letter abbreviations for compact UI
        team1_name = settings.team_names.team1_name
        team2_name = settings.team_names.team2_name

        abbr1 = team1_name[:3].upper()
        abbr2 = team2_name[:3].upper()

        # Centralize layout constants to keep overlay consistent across different resolutions
        MARGIN = 40
        CLOCK_W = 90
        SCORE_W = 260
        SCORE_H = 40
        GAP = 6

        POSBAR_H = 32
        POSBAR_GAP = 6

        total_width = CLOCK_W + GAP + SCORE_W

        x1 = MARGIN
        x2 = x1 + total_width

        score_y1 = MARGIN
        score_y2 = score_y1 + SCORE_H

        pos_y1 = score_y2 + POSBAR_GAP
        pos_y2 = pos_y1 + POSBAR_H

        # Render match time based on frame index to avoid relying on external timers
        time_sec = frame_num / float(fps)
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        clock_x1 = x1
        clock_x2 = x1 + CLOCK_W

        cv2.rectangle(frame, (clock_x1, score_y1), (clock_x2, score_y2), (245, 245, 245), -1)
        cv2.rectangle(frame, (clock_x1, score_y1), (clock_x2, score_y2), (200, 200, 200), 1)

        cv2.putText(
            frame, time_str,
            (clock_x1 + 8, score_y1 + SCORE_H // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA
        )

        # Draw a dark score bar with team-color stripes for immediate team recognition
        score_x1 = clock_x2 + GAP
        score_x2 = x2

        cv2.rectangle(frame, (score_x1, score_y1), (score_x2, score_y2), (255, 0, 0), -1)

        cv2.line(frame, (score_x1, score_y1), (score_x2, score_y1), (80, 110, 180), 1)
        cv2.line(frame, (score_x1, score_y2), (score_x2, score_y2), (20, 30, 70), 1)

        mid_y = score_y1 + SCORE_H // 2 + 8

        stripe_w = 14
        padding = 10
        text_offset = 14

        cv2.rectangle(
            frame,
            (score_x1 + padding, score_y1 + padding),
            (score_x1 + padding + stripe_w, score_y2 - padding),
            color1,
            -1
        )

        cv2.rectangle(
            frame,
            (score_x2 - padding - stripe_w, score_y1 + padding),
            (score_x2 - padding, score_y2 - padding),
            color2,
            -1
        )

        cv2.putText(
            frame, abbr1,
            (score_x1 + padding + stripe_w + text_offset, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

        (t2w, _), _ = cv2.getTextSize(abbr2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(
            frame, abbr2,
            (score_x2 - padding - stripe_w - text_offset - t2w, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Display the current score centrally to support match narrative overlays
        score_text = f"{score1}   {score2}"
        
        (stw, sth), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        sx = score_x1 + (SCORE_W - stw) // 2
        sy = score_y1 + SCORE_H // 2 + sth // 3

        cv2.putText(
            frame, score_text,
            (sx, sy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )

        center_line_x = score_x1 + SCORE_W // 2
        cv2.line(frame, (center_line_x, score_y1 + 8), (center_line_x, score_y2 - 8), (255, 255, 255), 1)

        # Render possession as a split bar to communicate sustained dominance at a glance
        pos_x1 = x1
        pos_x2 = x2

        cv2.rectangle(frame, (pos_x1, pos_y1), (pos_x2, pos_y2), (25, 25, 25), -1)

        bar_width = pos_x2 - pos_x1
        w1 = int(bar_width * pct1)

        cv2.rectangle(frame, (pos_x1, pos_y1), (pos_x1 + w1, pos_y2), color1, -1)
        cv2.rectangle(frame, (pos_x1 + w1, pos_y1), (pos_x2, pos_y2), color2, -1)

        cv2.line(frame, (pos_x1 + w1, pos_y1), (pos_x1 + w1, pos_y2), (255, 255, 255), 2)

        cv2.putText(
            frame, f"{pct1*100:.0f}%",
            (pos_x1 + 12, pos_y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        (p2w, _), _ = cv2.getTextSize(f"{pct2*100:.0f}%", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(
            frame, f"{pct2*100:.0f}%",
            (pos_x2 - 12 - p2w, pos_y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Optionally overlay short-lived match events (goals/shots) without breaking the base UI
        if event_text:
            text_scale = 1.5
            thickness = 3
            (tw, th), _ = cv2.getTextSize(event_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
            
            tx = (w - tw) // 2
            ty = h - 150 

            cv2.putText(frame, event_text, (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(frame, event_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        # Produce a new list of frames to keep input frames unmodified for reuse/debugging
        output_video_frame = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Read per-frame track dictionaries to decouple drawing from tracking logic
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]

            # Draw team-colored player markers and highlight current ball owner
            for track_id, player in player_dict.items():
                raw = player.get("team_color")
                color = (0,255,255) if raw is None else tuple(int(x) for x in np.asarray(raw).tolist())
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

            # Render referees separately to keep role semantics consistent in the UI
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            # Render goalkeepers with a distinct color to avoid confusion with outfield players
            for track_id, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255,0,0), track_id)
                
                if goalkeeper.get('has_ball', False):
                    frame = self.draw_triangle(frame, goalkeeper['bbox'], (0,0,255))
            
            # Render ball as a high-salience marker for quick visual tracking
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            # Add scoreboard and possession overlay last so it stays visible on top
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frame.append(frame)
        
        return output_video_frame
    
    def _get_referee_color(self, player_dict):
        # Default to a high-contrast style color, but switch when it would clash with black kits
        current_style_color = (0, 255, 255)
        black_style_color = (0, 0, 0)

        # Collect unique team colors from current players for conflict detection
        team_colors = []
        for p in player_dict.values():
            color = p.get("team_color")
            if color is not None:
                c = tuple(int(x) for x in color) if isinstance(color, np.ndarray) else color
                if c not in team_colors:
                    team_colors.append(c)

        # Detect a dark kit to avoid rendering referees in black on black
        is_conflict = False
        for c in team_colors:
            brightness = sum(c) / 3.0
            if brightness < 60:
                is_conflict = True
                break

        return current_style_color if is_conflict else black_style_color

    def _stabilize_roles_per_track(self, tracks: dict, min_observations: int = 5, min_ratio: float = 0.6) -> dict:
        # Exit early when tracks are empty to keep the pipeline fault-tolerant
        if not tracks or "players" not in tracks:
            return tracks

        num_frames = len(tracks["players"])
        if num_frames == 0:
            return tracks

        # Count per-track role observations to derive a stable majority assignment
        role_counts = {}

        for frame_idx in range(num_frames):
            for tid in tracks["players"][frame_idx].keys():
                role_counts.setdefault(tid, {"player": 0, "referee": 0, "goalkeeper": 0})
                role_counts[tid]["player"] += 1

            for tid in tracks["referees"][frame_idx].keys():
                role_counts.setdefault(tid, {"player": 0, "referee": 0, "goalkeeper": 0})
                role_counts[tid]["referee"] += 1

            for tid in tracks["goalkeepers"][frame_idx].keys():
                role_counts.setdefault(tid, {"player": 0, "referee": 0, "goalkeeper": 0})
                role_counts[tid]["goalkeeper"] += 1

        # Apply a minimum evidence threshold so short-lived tracks do not get forced roles
        global_role = {}

        for tid, counts in role_counts.items():
            total = counts["player"] + counts["referee"] + counts["goalkeeper"]
            if total < min_observations:
                continue

            role, count = max(counts.items(), key=lambda kv: kv[1])
            if count / total >= min_ratio:
                global_role[tid] = role

        # Rebuild tracks by routing each tid to its stable role bucket
        new_tracks = {
            "players":     [{} for _ in range(num_frames)],
            "referees":    [{} for _ in range(num_frames)],
            "goalkeepers": [{} for _ in range(num_frames)],
        }

        # Preserve non-role track keys (e.g., ball) unchanged
        for key, value in tracks.items():
            if key not in ("players", "referees", "goalkeepers"):
                new_tracks[key] = value

        for frame_idx in range(num_frames):
            for role_name in ("players", "referees", "goalkeepers"):
                for tid, track in tracks[role_name][frame_idx].items():
                    base_role = role_name[:-1]
                    final_role = global_role.get(tid, base_role)

                    if final_role == "player":
                        new_tracks["players"][frame_idx][tid] = track
                    elif final_role == "referee":
                        new_tracks["referees"][frame_idx][tid] = track
                    elif final_role == "goalkeeper":
                        new_tracks["goalkeepers"][frame_idx][tid] = track

        return new_tracks
    
    def get_object_tracks_from_video(self, video_path, read_from_stub=False, stub_path=None, batch_size=32, resume_from_stub=False, frame_skip: int = 1):
        # Prefer direct stub load for fast iteration when no resuming is requested
        if read_from_stub and not resume_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, frame_skip)
        effective_total_frames = ceil(total_frames / frame_skip)

        # Resume by jumping to the physical frame offset aligned with effective frame indexing
        if resume_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            already_processed_frames = len(tracks.get("players", []))
            cap.set(cv2.CAP_PROP_POS_FRAMES, already_processed_frames * frame_skip)
        else:
            tracks = {"players": [], "referees": [], "goalkeepers": [], "ball": []}
            already_processed_frames = 0

        # Track progress in effective frames to reflect frame_skip behavior
        pbar = tqdm(total=effective_total_frames, desc="YOLO tracking", initial=already_processed_frames)

        while True:
            # Build a batch of usable frames while skipping intermediate frames for throughput control
            frames_batch = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)

                for _ in range(frame_skip - 1):
                    ret_skip, _ = cap.read()
                    if not ret_skip:
                        break

            if not frames_batch:
                break

            # Run YOLO inference and update progress based on processed frames
            detections = self.detect_frames(frames_batch)
            pbar.update(len(detections))

            # Convert detections into the canonical per-frame track structure
            for detection in detections:
                frame_num = len(tracks["players"])

                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                detection_supervision = sv.Detections.from_ultralytics(detection)

                # Filter low-confidence balls to avoid noise in downstream possession and shot logic
                ball_class_id = cls_names_inv["ball"]
                ball_conf_min = 0.30

                mask_keep = []
                for cls_id, conf in zip(detection_supervision.class_id, detection_supervision.confidence):
                    mask_keep.append(not (cls_id == ball_class_id and conf < ball_conf_min))

                detection_supervision = detection_supervision[mask_keep]

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["goalkeepers"].append({})
                tracks["ball"].append({})

                # Store tracked entities by role to keep downstream logic simple and fast
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    if cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    if cls_id == cls_names_inv['goalkeeper']:
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

                # Persist a single ball bbox per frame using fixed track ID = 1
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0]
                    cls_id = frame_detection[3]
                    if cls_id == ball_class_id:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

        pbar.close()
        cap.release()

        # Apply role stabilization before caching to ensure consistent downstream semantics
        tracks = self._stabilize_roles_per_track(tracks)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_annotations_to_video(self, input_video_path, tracks, team_ball_control, output_path, fps=30, frame_skip: int = 1, settings=None):
        # Use a settings fallback to keep this function callable from standalone scripts
        if settings is None:
            from config import Settings
            settings = Settings()
            
        # Load pitch parameters once to keep per-frame loop minimal
        pitch_L = settings.analytics.pitch_length
        pitch_W = settings.analytics.pitch_width
        goal_W  = settings.analytics.goal_width
        goal_D  = settings.analytics.goal_depth
        shot_speed_limit = settings.analytics.shot_speed_threshold

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened(): 
            raise RuntimeError(f"Could not open video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, frame_skip)
        effective_total_frames = ceil(total_frames / frame_skip)
        
        # Maintain match state (score + event overlays) across frames
        score_team1 = 0
        score_team2 = 0
        goal_cooldown = 0
        
        current_event_text = None
        event_display_frames = 0
        
        prev_ball_pos_m = None
        
        frame_num = 0

        for _ in tqdm(range(effective_total_frames), desc="Rendering"):
            ret, frame = cap.read()
            if not ret: 
                break
            frame_copy = frame.copy()

            # Pull per-frame tracks defensively to handle partial stubs and boundary conditions
            player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
            ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
            referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
            goalkeeper_dict = tracks["goalkeepers"][frame_num] if frame_num < len(tracks["goalkeepers"]) else {}
            self.current_players = player_dict

            # Draw tracked entities first, then overlays, to ensure UI remains visible
            ref_color = self._get_referee_color(player_dict)
            for tid, p in player_dict.items():
                c = p.get("team_color", (0,255,255))
                if isinstance(c, np.ndarray): 
                    c = tuple(int(x) for x in c)
                frame_copy = self.draw_ellipse(frame_copy, p["bbox"], c, tid)
                if p.get("has_ball"): 
                    self.draw_triangle(frame_copy, p["bbox"], (0,0,255))
            
            for _, r in referee_dict.items(): 
                self.draw_ellipse(frame_copy, r["bbox"], ref_color)
            for tid, g in goalkeeper_dict.items(): 
                self.draw_ellipse(frame_copy, g["bbox"], (255,0,0), tid)

            # Convert ball bbox to pitch meters to support event heuristics and map overlays
            ball_m_x, ball_m_y = None, None
            
            if 1 in ball_dict:
                bbox = ball_dict[1]["bbox"]
                frame_copy = self.draw_triangle(frame_copy, bbox, (0, 255, 0))
                
                cx, cy = get_center_of_bbox(bbox)
                bmx, bmy = pixel_to_pitch(cx, cy)
                
                if np.isfinite(bmx) and np.isfinite(bmy):
                    ball_m_x, ball_m_y = bmx, bmy

            # Detect shots via high ball speed and directional movement near the attacking half
            if ball_m_x is not None and prev_ball_pos_m is not None:
                dx = ball_m_x - prev_ball_pos_m[0]
                dy = ball_m_y - prev_ball_pos_m[1]
                dist_m = (dx**2 + dy**2)**0.5
                
                dt = (1.0 / fps) * frame_skip
                speed_ms = dist_m / dt
                
                if speed_ms > shot_speed_limit and goal_cooldown == 0:
                    is_shooting_t1 = (dx > 0 and ball_m_x > pitch_L * 0.6)
                    is_shooting_t2 = (dx < 0 and ball_m_x < pitch_L * 0.4)
                    
                    if is_shooting_t1 or is_shooting_t2:
                        current_event_text = f"SHOT ({speed_ms*3.6:.0f} km/h)!"
                        event_display_frames = 15

            # Update ball cache only when coordinates are valid to prevent spikes
            if ball_m_x is not None:
                prev_ball_pos_m = (ball_m_x, ball_m_y)

            # Apply a cooldown to avoid counting the same goal across consecutive frames
            if goal_cooldown > 0:
                goal_cooldown -= 1
            
            # Detect goals by ball crossing the goal line within the goal mouth corridor
            if ball_m_x is not None and goal_cooldown == 0:
                goal_center_y = pitch_W / 2.0
                half_goal = goal_W / 2.0
                
                in_goal_y = (goal_center_y - half_goal) < ball_m_y < (goal_center_y + half_goal)
                
                if in_goal_y:
                    if ball_m_x < 0 and ball_m_x > -goal_D:
                        score_team2 += 1
                        current_event_text = "GOAL FOR TEAM 2!"
                        event_display_frames = 60
                        goal_cooldown = fps * 10
                        
                    elif ball_m_x > pitch_L and ball_m_x < (pitch_L + goal_D):
                        score_team1 += 1
                        current_event_text = "GOAL FOR TEAM 1!"
                        event_display_frames = 60
                        goal_cooldown = fps * 10

            # Decay event overlay state to keep the UI uncluttered
            text_to_show = None
            if event_display_frames > 0:
                text_to_show = current_event_text
                event_display_frames -= 1
            
            # Render scoreboard/possession with live score and optional event banner
            frame_copy = self.draw_team_ball_control(
                frame_copy, frame_num, team_ball_control, 
                score1=score_team1, score2=score_team2, 
                event_text=text_to_show, fps=fps
            )
            
            # Add a live pitch mini-map to support interpretability of the visual output
            frame_copy = self.draw_live_map(frame_copy, tracks, frame_num, settings)

            out.write(frame_copy)
            frame_num += 1

            # Skip physical frames to match effective frame indexing used by tracking
            for _ in range(frame_skip - 1):
                cap.read()

        cap.release()
        out.release()
        print(f"Saved annotated video to {output_path}")

    def _draw_voronoi_regions(self, map_img, players_data):
        # Require enough sites to make the Voronoi tessellation stable and informative
        if len(players_data) < 4:
            return map_img

        h, w = map_img.shape[:2]
        
        # Extract site coordinates for the Voronoi computation
        points = np.array([[p[0], p[1]] for p in players_data])
        
        # Add bounding dummy sites to close regions at the pitch border for fill operations
        dummy_points = [
            [-w, -h], [-w, 2*h], [2*w, -h], [2*w, 2*h],
            [-w, h//2], [2*w, h//2], [w//2, -h], [w//2, 2*h]
        ]
        points_all = np.vstack([points, np.array(dummy_points)])

        # Guard against numerical instability in degenerate configurations
        try:
            vor = Voronoi(points_all)
        except Exception:
            return map_img

        # Draw into an overlay to apply alpha blending and preserve pitch lines
        overlay = map_img.copy()

        # Fill each player's region using the player's team color as the territory identifier
        for i, region_idx in enumerate(vor.point_region):
            if i >= len(players_data):
                break
            
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            
            polygon = [vor.vertices[i] for i in region]
            polygon = np.array(polygon, dtype=np.int32)
            
            color = players_data[i][2]
            cv2.fillPoly(overlay, [polygon], color)

        # Blend overlay to keep the Voronoi result readable without overpowering the pitch texture
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, map_img, 1 - alpha, 0, map_img)
        
        return map_img

    def draw_live_map(self, frame, tracks, frame_idx, settings, alpha_smooth: float = 0.6):
        # Exit early when frame index exceeds available tracking data
        if frame_idx >= len(tracks["players"]):
            return frame

        # Lazy-init cached calibration, homography, pitch image and smoothing state for performance
        if not hasattr(self, "_live_map_initialized"):
            self._live_map_initialized = False
            
            pitch_full = cv2.imread(str(settings.paths.pitch_image))
            if pitch_full is None:
                print("[live_map] Pitch image not found:", settings.paths.pitch_image)
                return frame

            try:
                H_px = np.load(str(settings.paths.homography_npy))
                calib = np.load(str(settings.paths.calib_file))
                K = calib["K"]
                dist = calib["dist"]
            except Exception as e:
                print("[live_map] Loading error:", e)
                return frame

            full_h, full_w = pitch_full.shape[:2]
            
            # Resize once to a stable UI footprint that fits most 16:9 videos
            map_w, map_h = 350, 210 
            pitch_resized = cv2.resize(pitch_full, (map_w, map_h))

            # Cache expensive assets and stateful smoothers between frames
            self._live_map_H_px = H_px
            self._live_map_K = K
            self._live_map_dist = dist
            self._live_map_full_size = (full_w, full_h)
            self._live_map_img_base = pitch_resized
            self._live_map_size = (map_w, map_h)
            self._live_map_history = {"players": {}, "gks": {}, "refs": {}, "ball": None}
            self._live_map_initialized = True

        if not self._live_map_initialized:
            return frame

        H_px = self._live_map_H_px
        full_w, full_h = self._live_map_full_size
        map_w, map_h = self._live_map_size
        
        # Build two map layers to compare position-only vs territorial overlays side-by-side
        map_classic = self._live_map_img_base.copy()
        map_voronoi = self._live_map_img_base.copy()

        def cam_to_map_px(x, y):
            # Apply camera undistortion to align click space with the homography calibration
            pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
            undist_norm = cv2.undistortPoints(pts, self._live_map_K, self._live_map_dist)
            u, v = undist_norm[0, 0]
            x_u = self._live_map_K[0, 0] * u + self._live_map_K[0, 2]
            y_u = self._live_map_K[1, 1] * v + self._live_map_K[1, 2]
            pts_u = np.array([[[x_u, y_u]]], dtype=np.float32)
            
            # Transform into pitch image coordinates and reject out-of-bounds projections
            dst = cv2.perspectiveTransform(pts_u, H_px)
            X, Y = dst[0, 0]
            
            if X < 0 or X >= full_w or Y < 0 or Y >= full_h:
                return None

            return int(X / full_w * map_w), int(Y / full_h * map_h)

        def smooth(key, pos, store_dict):
            if pos is None: 
                return None
            new = np.array(pos, dtype=np.float32)
            if key in store_dict:
                old = store_dict[key]
                sm = alpha_smooth * new + (1.0 - alpha_smooth) * old
            else:
                sm = new
            store_dict[key] = sm
            return int(sm[0]), int(sm[1])

        hist = self._live_map_history
        voronoi_points = []

        # Draw players as colored markers and capture sites for Voronoi region filling
        players = tracks["players"][frame_idx]
        for pid, p in players.items():
            x1, y1, x2, y2 = p["bbox"]
            pos = cam_to_map_px(0.5*(x1+x2), y2)
            pos = smooth(pid, pos, hist["players"])
            if pos is None: 
                continue
            
            raw_color = p.get("team_color")
            color = (180, 180, 180) if raw_color is None else tuple(int(c) for c in np.asarray(raw_color).tolist())
            
            cv2.circle(map_classic, (pos[0], pos[1]), 7, (255, 255, 255), -1)
            cv2.circle(map_classic, (pos[0], pos[1]), 5, color, -1)
            
            voronoi_points.append((pos[0], pos[1], color))

        # Render goalkeepers with a distinct color and include them in territorial computation
        gks = tracks["goalkeepers"][frame_idx]
        for gid, g in gks.items():
            x1, y1, x2, y2 = g["bbox"]
            pos = cam_to_map_px(0.5*(x1+x2), y2)
            pos = smooth(gid, pos, hist["gks"])
            if pos is None: 
                continue
            
            color = (0, 200, 255)
            
            cv2.circle(map_classic, (pos[0], pos[1]), 8, (255, 255, 255), -1)
            cv2.circle(map_classic, (pos[0], pos[1]), 6, color, -1)
                
            voronoi_points.append((pos[0], pos[1], color))

        # Draw Voronoi background first, then re-draw markers to keep players visible
        map_voronoi = self._live_map_img_base.copy()
        map_voronoi = self._draw_voronoi_regions(map_voronoi, voronoi_points)
        
        for p in voronoi_points:
            cv2.circle(map_voronoi, (p[0], p[1]), 6, (255,255,255), 1)
            cv2.circle(map_voronoi, (p[0], p[1]), 4, p[2], -1)

        # Compute referee marker color dynamically to avoid black-on-black visibility issues
        current_players = tracks["players"][frame_idx]
        ref_map_color = (0, 0, 0)
        
        for p in current_players.values():
            c = p.get("team_color")
            if c is not None and (sum(c) / 3.0) < 60:
                ref_map_color = (0, 255, 255)
                break

        # Draw referees on both maps to keep role context in each visualization
        refs = tracks["referees"][frame_idx]
        for rid, r in refs.items():
            x1, y1, x2, y2 = r["bbox"]
            pos = cam_to_map_px(0.5*(x1+x2), y2)
            pos = smooth(rid, pos, hist["refs"])
            if pos is not None:
                for m in [map_classic, map_voronoi]:
                    cv2.circle(m, (pos[0], pos[1]), 5, (255, 255, 255), -1)
                    cv2.circle(m, (pos[0], pos[1]), 4, ref_map_color, -1)

        # Draw ball with smoothing to reduce jitter and maintain salience
        ball_dict = tracks["ball"][frame_idx]
        if 1 in ball_dict:
            x1, y1, x2, y2 = ball_dict[1]["bbox"]
            pos = cam_to_map_px(0.5*(x1+x2), 0.5*(y1+y2))
            if pos is not None:
                if hist["ball"] is None:
                    hist["ball"] = np.array(pos, dtype=np.float32)
                else:
                    new = np.array(pos, dtype=np.float32)
                    hist["ball"] = alpha_smooth * new + (1.0 - alpha_smooth) * hist["ball"]
                bx, by = int(hist["ball"][0]), int(hist["ball"][1])
                
                for m in [map_classic, map_voronoi]:
                    cv2.circle(m, (bx, by), 5, (0, 0, 0), -1)
                    cv2.circle(m, (bx, by), 3, (0, 255, 0), -1)

        # Compose the two maps with a separator to improve readability on the final video
        separator = np.zeros((map_h, 10, 3), dtype=np.uint8)
        combined_maps = np.hstack((map_classic, separator, map_voronoi))
        
        # Add labels directly on the sub-maps to guide interpretation during playback
        cv2.putText(map_classic, "Positions", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(map_voronoi, "Territory", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        total_w = map_w * 2 + 10
        total_h = map_h
        
        # Place the overlay near the bottom center to avoid covering key action areas
        h_frame, w_frame = frame.shape[:2]
        x_offset = max(0, (w_frame - total_w) // 2)
        y_offset = max(0, h_frame - total_h - 40)
        
        roi = frame[y_offset:y_offset+total_h, x_offset:x_offset+total_w]
        if roi.shape[:2] == combined_maps.shape[:2]:
            frame[y_offset:y_offset+total_h, x_offset:x_offset+total_w] = combined_maps
            cv2.rectangle(frame, (x_offset-2, y_offset-2), (x_offset+total_w+2, y_offset+total_h+2), (255,255,255), 2)

        return frame