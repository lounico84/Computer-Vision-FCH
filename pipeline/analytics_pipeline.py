from config import Settings
from analytics import export_frame_csv1


def export_analytics(tracks, team_ball_control, settings: Settings):
    """Export frame-level analytics to a csv file"""
    export_frame_csv1(
        tracks=tracks,
        team_ball_control=team_ball_control,
        fps=settings.tracking.fps, # use fps from config
        output_path=str(settings.paths.frame_events_csv), # output csv location from config
    )