from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    model_path: Path = Path("yolo_training/models/fifth_model/run1/weights/best.pt")
    input_video: Path = Path("input_videos_match/Test/kuesnacht_test_clip2.MP4")
    tracks_stub: Path = Path("project/Computer-Vision-FCH/stubs/track_stubs_k.pkl")
    output_video: Path = Path("output_video_match/output_video_k.avi")
    color_debug_image: Path = Path("output_video_match/color_debug_k.png")
    frame_events_csv: Path = Path("project/Computer-Vision-FCH/analytics/frame_events.csv")


@dataclass
class TrackingConfig:
    fps: int = 60
    read_tracks_from_stub: bool = True
    max_ball_interpolation_gap: int = 20


@dataclass
class RefereeDecisionConfig:
    min_votes: int = 7
    min_ratio: float = 0.8
    min_observations: int = 10


@dataclass
class BallControlConfig:
    min_switch_frames: int = 5  # Hysterese für Teamwechsel


@dataclass
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    referee: RefereeDecisionConfig = field(default_factory=RefereeDecisionConfig)
    ball_control: BallControlConfig = field(default_factory=BallControlConfig)