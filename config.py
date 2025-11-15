from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path("project/Computer-Vision-FCH")

# Centralized file path configuration
@dataclass
class PathConfig:
    # Model
    model_path: Path = "yolo_training/models/fifth_model/run1/weights/best.pt"

    # Videos
    input_video: Path = "input_videos_match/Test/kuesnacht_test_clip2.MP4"
    test_input_video: Path = "input_videos_match/Test/wiesendangen_test_clip_short.mp4"

    # Output
    output_video: Path = "output_video_match/output_video.avi"
    color_debug_image: Path = "output_video_match/color_debug.png"

    # Stubs
    tracks_stub: Path = PROJECT_ROOT / "stubs/track_stubs.pkl"

    # Analytics
    frame_events_csv: Path = PROJECT_ROOT / "analytics/frame_events.csv"
    pass_map_team1: Path = PROJECT_ROOT / "analytics/pass_map_team1.png"
    pass_map_team2: Path = PROJECT_ROOT / "analytics/pass_map_team2.png"

    # Calibration
    pitch_image: Path = PROJECT_ROOT / "calibration/fch_fussballfeld.jpg"
    homography_npy: Path = PROJECT_ROOT / "calibration/aio_homography_cam_to_map.npy"
    homography_npz: Path = PROJECT_ROOT / "calibration/homography.npz"
    calib_file: Path = PROJECT_ROOT / "calibration/aio_gopro_calib_approx.npz"
    warped_frame_output: Path = PROJECT_ROOT / "calibration/aio_warped_frame_to_pitch.png"

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