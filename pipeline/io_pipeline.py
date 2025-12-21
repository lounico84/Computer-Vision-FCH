from config import Settings
from utils import read_video, save_video
from trackers import Tracker


# Load tracking outputs by running the detector/tracker on the configured video or reusing a cached stub
def load_video_and_tracks(settings: Settings):

    # Pull path and tracking parameters from the central Settings object
    paths = settings.paths
    tracking_cfg = settings.tracking

    # Initialize the tracker with the configured YOLO weights to ensure consistent detections
    tracker = Tracker(str(paths.model_path))

    # Generate tracks directly from the video stream or load from stub to reduce runtime on repeats
    tracks = tracker.get_object_tracks_from_video(
        str(paths.input_video),
        read_from_stub=tracking_cfg.read_tracks_from_stub,
        stub_path=str(paths.tracks_stub),
        resume_from_stub=tracking_cfg.resume_track_from_stub if hasattr(tracking_cfg, "resume_track_from_stub") else False,
        frame_skip=tracking_cfg.frame_skip if hasattr(tracking_cfg, "frame_skip") else 1,
    )

    # Smooth ball trajectory by interpolating short gaps to improve downstream event detection stability
    if tracking_cfg.max_ball_interpolation_gap > 0:
        tracks["ball"] = tracker.interpolate_ball_positions(
            tracks["ball"],
            max_gap=tracking_cfg.max_ball_interpolation_gap,
            max_jump_px=80,  # Tune to trade off between recovery and false bridges
        )

    # Return only the tracker and track data to keep the pipeline memory-efficient
    return tracker, tracks