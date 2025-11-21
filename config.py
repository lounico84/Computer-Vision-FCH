from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Centralized file path configuration
@dataclass
class PathConfig:
    # Model
    model_path: Path = "yolo_training/models/fifth_model/run1/weights/best.pt"

    # Videos
    input_video: Path = "input_videos_match/Test/kuesnacht_test_clip4.MP4"

    # Output
    output_video: Path = "output_video_match/output_video_k_4_30.avi"
    color_debug_image: Path = "output_video_match/color_debug_4_30.png"

    # Stubs
    tracks_stub: Path = PROJECT_ROOT / "stubs/track_stubs_k_4_30.pkl"
    team_stub: Path = PROJECT_ROOT / "stubs/team_stubs_k_4_30.pkl"

    # Analytics
    frame_events_csv: Path = PROJECT_ROOT / "analytics/frame_events/frame_events_4_30.csv"
    pass_map_team1: Path = PROJECT_ROOT / "analytics/pass_maps/pass_map_team1_4_32.png"
    pass_map_team2: Path = PROJECT_ROOT / "analytics/pass_maps/pass_map_team2_4_32.png"

    # Calibration
    pitch_image: Path = PROJECT_ROOT / "calibration/pictures/fch_fussballfeld.jpg"
    homography_npy: Path = PROJECT_ROOT / "calibration/data/aio_homography_cam_to_map.npy"
    homography_npz: Path = PROJECT_ROOT / "calibration/data/homography.npz"
    calib_file: Path = PROJECT_ROOT / "calibration/data/aio_gopro_calib_approx.npz"
    warped_frame_output: Path = PROJECT_ROOT / "calibration/pictures/aio_warped_frame_to_pitch.png"

# Configuration for model inference and tracking behavior
@dataclass
class TrackingConfig:
    fps: int = 30
    read_tracks_from_stub: bool = True      # read current yolo predictions and don't predict again
    resume_track_from_stub: bool = True     # continue with frames for training
    read_team_from_stub: bool = False       # read current k-means team assignments
    resume_team_from_stub: bool = True      # continue with frames for training
    max_ball_interpolation_gap: int = 20
    frame_skip: int = 2

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

# Configuration of pitch and ball statistics
@dataclass
class AnalyticsConfig:
    pitch_length: float = 100.0 
    pitch_width: float = 60.0     
    pass_speed_threshold: float = 2.0
    pass_min_distance: float = 7.0
    max_ball_speed: float = 40.0      # 144 km/h
    pitch_margin: float = 3.0
    pass_min_frames: int = 3

# Global application settings
@dataclass
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    referee: RefereeDecisionConfig = field(default_factory=RefereeDecisionConfig)
    ball_control: BallControlConfig = field(default_factory=BallControlConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)