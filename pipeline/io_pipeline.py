from config import Settings
from utils import read_video, save_video
from trackers import Tracker


def load_video_and_tracks(settings: Settings):
    paths = settings.paths
    tracking_cfg = settings.tracking

    video_frames = read_video(str(paths.input_video))

    tracker = Tracker(str(paths.model_path))

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=tracking_cfg.read_tracks_from_stub,
        stub_path=str(paths.tracks_stub),
    )

    # Interpolation of Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(
        tracks["ball"],
        max_gap=tracking_cfg.max_ball_interpolation_gap,
    )

    return video_frames, tracker, tracks


def render_and_save_video(video_frames, tracks, team_ball_control, tracker, settings: Settings):
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    save_video(output_frames, str(settings.paths.output_video))