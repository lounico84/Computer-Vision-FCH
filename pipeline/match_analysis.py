import gc
from config import Settings
from pipeline.io_pipeline import load_video_and_tracks
from pipeline.team_pipeline import assign_teams, assign_goalkeepers_to_teams
from pipeline.ball_pipeline import compute_team_ball_control
from pipeline.analytics_pipeline import export_analytics

def run_match_analysis(settings: Settings | None = None):

    # Use default settings object if none is provided
    if settings is None:
        settings = Settings()

    paths = settings.paths
    tracking = settings.tracking

    # Load tracking information
    print("[STEP 1] - loading tracking information...")
    tracker, tracks = load_video_and_tracks(settings)

    # Team classification and goalkeeper assignment
    print("[STEP 2] - team classification and goalkeeper assignment...")
    tracks, team_assigner = assign_teams(tracks, settings, stub_path=paths.team_stub, read_from_stub=True, save_stub=True, frame_skip=tracking.frame_skip)
    tracks = assign_goalkeepers_to_teams(tracks, team_assigner)

    print("\n[STEP 3] - computing team ball controll over all frames...")
    # Compute team ball control over all frames
    team_ball_control = compute_team_ball_control(tracks, settings)

    print("[STEP 4] - drawing anntoations...")
    # Annotiertes Video direkt streamend schreiben
    tracker.draw_annotations_to_video(
        str(paths.input_video),
        tracks,
        team_ball_control,
        output_path=str(paths.output_video),
        fps=tracking.fps,
        frame_skip=tracking.frame_skip
    )

    print("\n[STEP 5] - exporting analytics...")
    # Export analytics
    export_analytics(tracks, team_ball_control, settings)

    # KEIN render_and_save_video mehr