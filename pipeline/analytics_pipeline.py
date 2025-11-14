from config import Settings
from analytics import export_frame_csv1, create_pass_maps_from_csv

# Export frame-level analytics to a csv file
def export_analytics(tracks, team_ball_control, settings: Settings):

    export_frame_csv1(
        tracks=tracks,
        team_ball_control=team_ball_control,
        fps=settings.tracking.fps, # use fps from config
        output_path=str(settings.paths.frame_events_csv), # output csv location from config
    )

    # 3) Pass-Maps
    create_pass_maps_from_csv(
        csv_path=str(settings.paths.frame_events_csv),
        out_path_team1=str(settings.paths.pass_map_team1),
        out_path_team2=str(settings.paths.pass_map_team2),
        pitch_length=105.0,
        pitch_width=68.0,
        fps=settings.tracking.fps,
        speed_threshold=5.0,   # hier kannst du später spielen
        min_distance=1.5,
    )