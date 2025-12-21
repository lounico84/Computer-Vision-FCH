from config import Settings
from analytics import export_frame_csv1, create_pass_maps_from_csv


# Export frame-level tracking data and derived analytics artifacts based on the project configuration
def export_analytics(tracks, team_ball_control, settings: Settings):

    # Access analytics-specific parameters from the central configuration
    analytics_cfg = settings.analytics

    # Persist per-frame events (ball position, ownership, speed, counts) to a CSV for downstream analysis
    export_frame_csv1(
        tracks=tracks,
        team_ball_control=team_ball_control,
        fps=settings.tracking.fps,                 # Use FPS defined in tracking configuration
        output_path=str(settings.paths.frame_events_csv),  # Write CSV to configured output path
    )

    # Optional: Generate per-team pass maps from the exported frame-level CSV
    '''
    create_pass_maps_from_csv(
        csv_path=str(settings.paths.frame_events_csv),
        out_path_team1=str(settings.paths.pass_map_team1),
        out_path_team2=str(settings.paths.pass_map_team2),
        pitch_length=analytics_cfg.pitch_length,
        pitch_width=analytics_cfg.pitch_width,
        fps=settings.tracking.fps,
        speed_threshold=analytics_cfg.pass_speed_threshold,
        min_distance=analytics_cfg.pass_min_distance,
        pitch_image_path=str(settings.paths.pitch_image),
    )
    '''