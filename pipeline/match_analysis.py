from config import Settings
from pipeline.io_pipeline import load_video_and_tracks, render_and_save_video
from pipeline.team_pipeline import assign_teams, assign_goalkeepers_to_teams
from pipeline.ball_pipeline import compute_team_ball_control
from pipeline.analytics_pipeline import export_analytics


def run_match_analysis(settings: Settings | None = None):
    if settings is None:
        settings = Settings()

    # Video + Tracks
    video_frames, tracker, tracks = load_video_and_tracks(settings)

    # Teams / Ref / Keeper
    tracks, team_assigner = assign_teams(video_frames, tracks, settings)
    tracks = assign_goalkeepers_to_teams(tracks, team_assigner)

    # Team-Ball Control
    team_ball_control = compute_team_ball_control(tracks, settings)

    # Analytics
    export_analytics(tracks, team_ball_control, settings)

    # Visualisation
    render_and_save_video(video_frames, tracks, team_ball_control, tracker, settings)