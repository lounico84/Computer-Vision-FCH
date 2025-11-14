from config import Settings
from analytics import export_frame_csv1


def export_analytics(tracks, team_ball_control, settings: Settings):
    """Exportiert Frame-Statistiken als CSV."""
    export_frame_csv1(
        tracks=tracks,
        team_ball_control=team_ball_control,
        fps=settings.tracking.fps,
        output_path=str(settings.paths.frame_events_csv),
    )