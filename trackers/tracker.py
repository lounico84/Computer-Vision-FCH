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
from utils import get_bbox_width, get_center_of_bbox, get_bbox_height

class Tracker:
    def __init__(self, model_path):
        # Using GPU
        #device = torch.device("mps")
        #self.model = YOLO(model_path).to(device) # load YOLO model from the given path

        # Using CPU
        self.model = YOLO(model_path) # load YOLO model from the given path
        self.tracker = sv.ByteTrack() # initialize ByteTrack tracker from supervision
    
    # Interpolate missing ball positions over time
    def interpolate_ball_positions(self, ball_positions, max_gap=20, max_jump_px=80):
        raw = []
        for x in ball_positions:
            if 1 in x:
                raw.append(x[1]['bbox'])
            else:
                raw.append([np.nan, np.nan, np.nan, np.nan])

        df = pd.DataFrame(raw, columns=['x1','y1','x2','y2'])

        # Calculate center
        cx = (df["x1"] + df["x2"]) / 2.0
        cy = (df["y1"] + df["y2"]) / 2.0

        # Mark unplausabel jumps as NaN
        prev_cx = cx.shift(1)
        prev_cy = cy.shift(1)
        dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

        # If jump > max_jump_px, it dosen't count
        mask_big_jump = dist > max_jump_px
        df.loc[mask_big_jump, ["x1","y1","x2","y2"]] = np.nan

        # # Interpolate only gaps up to max_gap frames (for longer gaps keep NaN)
        df = df.interpolate(limit=max_gap, limit_direction="both")

        ball_tracks = []
        for bbox in df.to_numpy().tolist():
            if any(np.isnan(bbox)):
                # No reliable ball position in this frame
                ball_tracks.append({})
            else:
                # Use track ID 1 as the ball
                ball_tracks.append({1: {"bbox": bbox}})
        return ball_tracks

    # Run YOLO+tracking on the video in batches
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range (0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i+batch_size],
                conf=0.1,
                verbose=False,
                )
            detections += detections_batch
        
        # Returns a list of detection/tracking results, one per frame
        return detections

    # Get tracked objects (players, referees, goalkeepers, ball) for all frames
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        tracks = None

        # Check if pickle file exists to not run yolo again for developement
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)

        else:
            # Run detection + tracking over the full video
            detections = self.detect_frames(frames)

            # Initialize track data structure
            # Each list has one dict per frame
            # Contains bbox, class and tracker
            tracks={
                "players":[],
                "referees":[],
                "goalkeepers":[],
                "ball":[]
            }

            for frame_num, detection in enumerate(detections):
                cls_names = detection.names
                # Invert mapping: name to class_id
                cls_names_inv = {v:k for k,v in cls_names.items()}

                # Convert to Supervision detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)

                # Detect balls with higher confidence
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}
                ball_class_id = cls_names_inv["ball"]

                ball_conf_min = 0.30

                mask_keep = []
                for cls_id, conf in zip(detection_supervision.class_id,
                                        detection_supervision.confidence):
                    if cls_id == ball_class_id and conf < ball_conf_min:
                        mask_keep.append(False)
                    else:
                        mask_keep.append(True)
        
                detection_supervision = detection_supervision[mask_keep]

                # Track Objects with ByteTrack to get consistent track IDs over time
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                # Prepare empty containers for this frame
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["goalkeepers"].append({})
                tracks["ball"].append({})

                # Add tracked objects (players, referees, goalkeepers)
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox":bbox}

                    if cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                    
                    if cls_id == cls_names_inv['goalkeeper']:
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}

                # Add ball detections (always ID=1)
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0]
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv['ball']:
                        tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        # NEW: stabilize roles per track_id across the whole video
        tracks = self._stabilize_roles_per_track(
            tracks,
            min_observations=5,
            min_ratio=0.6,
        )

        # Store the tracks in a pickle file for faster debugging
        # Only save when we actually computed them, not when reading from stub
        if stub_path is not None and not read_from_stub:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    # Draw an ellipse under a player/referee/goalkeeper
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        height = get_bbox_height(bbox)

        # Draw the ellipse at the player's feet
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

        # ============================
        # Dynamic Track-ID Box Size
        # ============================

        # Scale box clearly relative to player size
        # Höhe stärker gewichten, damit Spieler vorne/hinten deutlich unterschiedliche Boxgrößen haben
        rectangle_height = int(max(height * 0.25, 10))   # min 10px, ca. 25% der BBox-Höhe
        rectangle_width = int(max(width * 0.70, 25))     # min 25px, ca. 70% der BBox-Breite

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2

        # Move rectangle slightly above the ellipse
        y1_rect = y2 + 8
        y2_rect = y1_rect + rectangle_height

        if track_id is not None:
            # Draw label box
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            # ======================
            # Dynamic Text Size
            # ======================
            font_scale = max(rectangle_height / 22, 0.4)   # skaliert stärker und hat eine Untergrenze
            text_y = y1_rect + int(rectangle_height * 0.75)

            # Center text horizontally
            text = f"{track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x_center - text_size[0] // 2

            # Put Track ID in box
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
    
    # Draw a small triangle above the bounding box
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        traingle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [traingle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traingle_points], 0, (0,0,0), 2)

        return frame

    # Draw a small overlay for ball position
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Use data from frame 0 up to and including the current frame
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Only consider frames where Team 1 or Team 2 actually had possession
        # (0 or other values mean: no clear control)
        mask_1 = (team_ball_control_till_frame == 1)
        mask_2 = (team_ball_control_till_frame == 2)

        team_1_num_frames = mask_1.sum()
        team_2_num_frames = mask_2.sum()
        denom = team_1_num_frames + team_2_num_frames

        if denom == 0:
             # Start with no ball control
            team_1 = 0.0
            team_2 = 0.0
        else:
            team_1 = team_1_num_frames / denom
            team_2 = team_2_num_frames / denom

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0, 0),
            3
        )

        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0, 0),
            3
        )

        return frame
    
    # Draw all annotations (players, referees, goalkeepers, ball, and team ball control stats) for every frame in the video.
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frame = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                raw = player.get("team_color")
                color = (0,255,255) if raw is None else tuple(int(x) for x in np.asarray(raw).tolist())
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # Highlight players who currently have the ball
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            # Draw Goalkeepers
            for track_id, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255,0,0), track_id)
                
                # Highlight goalkeeper who currently has the ball
                if goalkeeper.get('has_ball', False):
                    frame = self.draw_triangle(frame, goalkeeper['bbox'], (0,0,255))
            
            # Draw Ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frame.append(frame)
        
        # Returns a list of annotated frames to be written to a video file
        return output_video_frame
    
    # Stabalizes track_id: every track_id is assigned to one role player, goalkeeper or referee
    def _stabilize_roles_per_track(
        self,
        tracks: dict,
        min_observations: int = 5,
        min_ratio: float = 0.6,
    ) -> dict:

        # Falls keine Daten o.ä. – einfach unverändert zurückgeben
        if not tracks or "players" not in tracks:
            return tracks

        num_frames = len(tracks["players"])
        if num_frames == 0:
            return tracks

        # 1) Rollen-Votes pro track_id sammeln
        role_counts = {}  # track_id -> {"player": n, "referee": n, "goalkeeper": n}

        for frame_idx in range(num_frames):
            # players
            for tid in tracks["players"][frame_idx].keys():
                if tid not in role_counts:
                    role_counts[tid] = {"player": 0, "referee": 0, "goalkeeper": 0}
                role_counts[tid]["player"] += 1

            # referees
            for tid in tracks["referees"][frame_idx].keys():
                if tid not in role_counts:
                    role_counts[tid] = {"player": 0, "referee": 0, "goalkeeper": 0}
                role_counts[tid]["referee"] += 1

            # goalkeepers
            for tid in tracks["goalkeepers"][frame_idx].keys():
                if tid not in role_counts:
                    role_counts[tid] = {"player": 0, "referee": 0, "goalkeeper": 0}
                role_counts[tid]["goalkeeper"] += 1

        # 2) Für jede track_id eine Mehrheits-Rolle bestimmen
        global_role = {}  # track_id -> "player" / "referee" / "goalkeeper"

        for tid, counts in role_counts.items():
            total = counts["player"] + counts["referee"] + counts["goalkeeper"]
            if total < min_observations:
                # zu wenig Info -> keine globale Entscheidung, wir lassen YOLO-Flickern bestehen
                continue

            # dominante Rolle finden
            role, count = max(counts.items(), key=lambda kv: kv[1])
            if count / total >= min_ratio:
                # klare Mehrheit -> globale Rolle festlegen
                global_role[tid] = role
            # else: keine klare Mehrheit -> auch hier nichts erzwingen

        # 3) Neue Track-Struktur mit konsistenten Rollen aufbauen
        new_tracks = {
            "players":     [ {} for _ in range(num_frames) ],
            "referees":    [ {} for _ in range(num_frames) ],
            "goalkeepers": [ {} for _ in range(num_frames) ],
        }

        # Ball (und evtl. andere Keys) unverändert übernehmen
        for key, value in tracks.items():
            if key not in ("players", "referees", "goalkeepers"):
                new_tracks[key] = value

        for frame_idx in range(num_frames):
            for role_name in ("players", "referees", "goalkeepers"):
                for tid, track in tracks[role_name][frame_idx].items():
                    # Standard: Rolle so lassen, wie sie in diesem Frame war
                    base_role = role_name[:-1]  # "players" -> "player", "referees" -> "referee", ...
                    final_role = global_role.get(tid, base_role)

                    if final_role == "player":
                        new_tracks["players"][frame_idx][tid] = track
                    elif final_role == "referee":
                        new_tracks["referees"][frame_idx][tid] = track
                    elif final_role == "goalkeeper":
                        new_tracks["goalkeepers"][frame_idx][tid] = track

        return new_tracks
    
    def get_object_tracks_from_video(self,
                                     video_path,
                                     read_from_stub=False,
                                     stub_path=None,
                                     batch_size=32,
                                     resume_from_stub=False,
                                     frame_skip: int = 1
    ):
        """
        Speicher-schonende Variante:
        - liest das Video frame-weise mit cv2.VideoCapture
        - verarbeitet Frames in Batches
        - hält NIE das ganze Video im RAM
        """

        # Falls Stub genutzt werden soll und existiert: direkt laden
    def get_object_tracks_from_video(
    self,
    video_path,
    read_from_stub=False,
    stub_path=None,
    batch_size=32,
    resume_from_stub=False,
    frame_skip: int = 1,
    ):
        """
        Speicher-schonende Variante:
        - liest das Video frame-weise mit cv2.VideoCapture
        - verarbeitet Frames in Batches
        - hält NIE das ganze Video im RAM
        - optional: kann von einem vorhandenen Stub (.pkl) weitermachen (resume_from_stub=True)
        - frame_skip: nur jede n-te Frame verarbeiten (z.B. 2 => jede 2. Frame)
        """

        # Fall 1: altes Verhalten – kompletten Stub nur laden und zurückgeben
        if read_from_stub and not resume_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_skip < 1:
            frame_skip = 1

        # effektive Anzahl verarbeiteter Frames (z.B. 60fps, frame_skip=2 => ~30fps)
        effective_total_frames = ceil(total_frames / frame_skip)

        # Fall 2: Von vorhandenen Tracks weiterrechnen (Resume)
        if resume_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            # Anzahl bereits bearbeiteter (effektiver) Frames bestimmen
            already_processed_frames = len(tracks.get("players", []))
            # Video an entsprechende physische Frameposition springen
            physical_start_frame = already_processed_frames * frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, physical_start_frame)
        else:
            # Neu von vorne starten
            tracks = {
                "players": [],
                "referees": [],
                "goalkeepers": [],
                "ball": [],
            }
            already_processed_frames = 0

        # Progressbar startet bei effective_initial
        pbar = tqdm(total=effective_total_frames, desc="YOLO tracking", initial=already_processed_frames)

        while True:
            # 1) Batch von Frames einlesen (nur jede frame_skip-te)
            frames_batch = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)

                # Zusätzliche Frames zwischen den Nutz-Frames überspringen
                for _ in range(frame_skip - 1):
                    ret_skip, _ = cap.read()
                    if not ret_skip:
                        break

            if not frames_batch:
                break  # Video zu Ende

            # 2) YOLO+Tracking auf diesen Batch
            detections = self.detect_frames(frames_batch)  # deine bestehende Funktion

            pbar.update(len(detections))

            # 3) Für jede Detection einen neuen Frame-Eintrag in tracks anlegen
            for detection in detections:
                frame_num = len(tracks["players"])

                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                detection_supervision = sv.Detections.from_ultralytics(detection)

                # Ball-Filtern wie in get_object_tracks
                ball_class_id = cls_names_inv["ball"]
                ball_conf_min = 0.30

                mask_keep = []
                for cls_id, conf in zip(detection_supervision.class_id,
                                        detection_supervision.confidence):
                    if cls_id == ball_class_id and conf < ball_conf_min:
                        mask_keep.append(False)
                    else:
                        mask_keep.append(True)

                detection_supervision = detection_supervision[mask_keep]

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                # Leere Dicts für diesen Frame
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["goalkeepers"].append({})
                tracks["ball"].append({})

                # Spieler / Schiri / Goalie eintragen
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

                # Ball (immer ID=1)
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0]
                    cls_id = frame_detection[3]
                    if cls_id == ball_class_id:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

        pbar.close()
        cap.release()

        # Rollen stabilisieren wie gehabt
        tracks = self._stabilize_roles_per_track(tracks)

        # Optional Stub speichern
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_annotations_to_video(self, input_video_path, tracks, team_ball_control, output_path, fps=30, frame_skip: int = 1):
        """
        Liest das Input-Video frame-weise,
        zeichnet die Overlays und schreibt direkt in eine Ausgabedatei.
        Kein Sammeln aller Frames im RAM.
        """

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_video_path}")

        # Videogrösse ermitteln
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_skip < 1:
            frame_skip = 1

        effective_total_frames = ceil(total_frames / frame_skip)
        frame_num = 0

        for _ in tqdm(range(effective_total_frames), desc="Progress"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_copy = frame.copy()

            player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
            ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
            referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
            goalkeeper_dict = tracks["goalkeepers"][frame_num] if frame_num < len(tracks["goalkeepers"]) else {}

            # Draw Players
            for track_id, player in player_dict.items():
                raw = player.get("team_color")
                color = (0,255,255) if raw is None else tuple(int(x) for x in np.asarray(raw).tolist())
                frame_copy = self.draw_ellipse(frame_copy, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame_copy = self.draw_triangle(frame_copy, player['bbox'], (0,0,255))

            # Draw Referees
            for _, referee in referee_dict.items():
                frame_copy = self.draw_ellipse(frame_copy, referee["bbox"], (0,255,255))

            # Draw Goalkeepers
            for track_id, goalkeeper in goalkeeper_dict.items():
                frame_copy = self.draw_ellipse(frame_copy, goalkeeper["bbox"], (255,0,0), track_id)
                if goalkeeper.get('has_ball', False):
                    frame_copy = self.draw_triangle(frame_copy, goalkeeper['bbox'], (0,0,255))

            # Draw Ball
            for _, ball in ball_dict.items():
                frame_copy = self.draw_triangle(frame_copy, ball["bbox"], (0,255,0))

            # Draw Team Ball Control
            frame_copy = self.draw_team_ball_control(frame_copy, frame_num, team_ball_control)

            out.write(frame_copy)
            frame_num += 1

            # Zwischengelagerte Frames überspringen
            for _ in range(frame_skip - 1):
                ret_skip, _ = cap.read()
                if not ret_skip:
                    break

        cap.release()
        out.release()
        print(f"Saved annotated video to {output_path}")