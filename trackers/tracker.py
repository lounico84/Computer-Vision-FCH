from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import numpy as np
import pandas as pd
import os
from utils import get_bbox_width, get_center_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) # load YOLO model from the given path
        self.tracker = sv.ByteTrack() # initialize ByteTrack tracker from supervision
    
    # Interpolate missing ball positions over time
    def interpolate_ball_positions(self, ball_positions, max_gap=20, max_jump_px=60):
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
            detections_batch = self.model.track(
                frames[i:i+batch_size],
                conf=0.1,
                )
            detections += detections_batch
        
        # Returns a list of detection/tracking results, one per frame
        return detections

    # Get tracked objects (players, referees, goalkeepers, ball) for all frames
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # Check if pickle file exists to not run yolo again for developement
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
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

            # Convert Goalkeeper to Player object
            '''
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            '''

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

        # Store the tracks in a pickle file for faster debugging
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    # Draw an ellipse under a player/referee/goalkeeper
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw the ellipse at the player's feet
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Create a small rectangle above the ellipse for the track ID
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            # Adjust text if ID is > 99
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10
            
            # Put the track ID number inside the rectangle
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0,0,0),
                2
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