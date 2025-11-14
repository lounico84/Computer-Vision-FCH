from config import Settings
from utils import read_video, save_video
from trackers import Tracker

# Load the input video and generate or load object tracking data
def load_video_and_tracks(settings: Settings):

    paths = settings.paths
    tracking_cfg = settings.tracking

    # Read the entire video into a list of frames
    video_frames = read_video(str(paths.input_video))

    # Initialize the tracker with the configured YOLO model
    tracker = Tracker(str(paths.model_path))

    # Retrieve tracking data (players, goalkeepers, referees, ball)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=tracking_cfg.read_tracks_from_stub,
        stub_path=str(paths.tracks_stub), # if tracking results of the video already exists, load them to save computation time
    )

    # Fill in missing ball positions across frames
    # Reduces flickering when the ball is temporarily not detected
    tracks["ball"] = tracker.interpolate_ball_positions(
        tracks["ball"],
        max_gap=tracking_cfg.max_ball_interpolation_gap,
    )

    return video_frames, tracker, tracks

# Draw visual annotations one each frame and save the final output video
def render_and_save_video(video_frames, tracks, team_ball_control, tracker, settings: Settings):
    
    # Add visual overlays to each frame
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save the anntoated video to the configured output path
    save_video(output_frames, str(settings.paths.output_video))