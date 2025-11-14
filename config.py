from dataclasses import dataclass, field
from pathlib import Path

# Centralized file path configuration
@dataclass
class PathConfig:
    model_path: Path = Path("yolo_training/models/fifth_model/run1/weights/best.pt")
    input_video: Path = Path("input_videos_match/Test/kuesnacht_test_clip2.MP4")
    tracks_stub: Path = Path("stubs/track_stubs_k.pkl")
    output_video: Path = Path("output_video_match/output_video_k.avi")
    color_debug_image: Path = Path("output_video_match/color_debug_k.png")
    frame_events_csv: Path = Path("project/Computer-Vision-FCH/analytics/frame_events.csv")
    pass_map_team1: Path = Path("project/Computer-Vision-FCH/analytics/pass_map_team1.png")
    pass_map_team2: Path = Path("project/Computer-Vision-FCH/analytics/pass_map_team2.png")

# Configuration for model inference and tracking behavior
@dataclass
class TrackingConfig:
    fps: int = 60
    read_tracks_from_stub: bool = True
    max_ball_interpolation_gap: int = 20

# Thresholds used to classify whether a player is actually a referee
@dataclass
class RefereeDecisionConfig:
    min_votes: int = 7
    min_ratio: float = 0.8
    min_observations: int = 10

# Configuration for smoothing ball possession decisions
@dataclass
class BallControlConfig:
    min_switch_frames: int = 5

# Global application settings
@dataclass
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    referee: RefereeDecisionConfig = field(default_factory=RefereeDecisionConfig)
    ball_control: BallControlConfig = field(default_factory=BallControlConfig)