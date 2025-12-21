import gc
from config import Settings
from pipeline.io_pipeline import load_video_and_tracks
from pipeline.team_pipeline import assign_teams, assign_goalkeepers_to_teams
from pipeline.ball_pipeline import compute_team_ball_control
from pipeline.analytics_pipeline import export_analytics
from pipeline.report_pipeline import build_pdf_report_from_notebook


# Orchestrate the full match analytics pipeline from tracking to video export and report generation
def run_match_analysis(settings: Settings | None = None):

    # Fall back to default configuration to ensure consistent paths and parameters
    if settings is None:
        settings = Settings()

    paths = settings.paths
    tracking = settings.tracking

    # Load or generate object tracks (players, goalkeepers, ball) as the pipeline baseline
    print("[STEP 1] - loading tracking information...")
    tracker, tracks = load_video_and_tracks(settings)

    # Assign teams and goalkeepers to support team-level analytics and visual overlays
    print("[STEP 2] - team classification and goalkeeper assignment...")
    tracks, team_assigner = assign_teams(
        tracks,
        settings,
        stub_path=paths.team_stub,
        read_from_stub=tracking.read_team_from_stub,
        save_stub=True,
        frame_skip=tracking.frame_skip,
        resume_from_stub=tracking.resume_team_from_stub,
    )
    tracks = assign_goalkeepers_to_teams(tracks, team_assigner)

    # Compute a per-frame possession signal used by event detection and broadcast-style overlays
    print("\n[STEP 3] - computing team ball controll over all frames...")
    team_ball_control = compute_team_ball_control(tracks, settings)

    # Stream annotated output video to disk to avoid holding full frame buffers in memory
    print("[STEP 4] - drawing anntoations...")
    tracker.draw_annotations_to_video(
        str(paths.input_video),
        tracks,
        team_ball_control,
        output_path=str(paths.output_video),
        fps=tracking.fps,
        frame_skip=tracking.frame_skip,
        settings=settings,
    )

    # Export frame-level analytics artifacts and compile the PDF report for delivery
    print("\n[STEP 5] - exporting analytics...")
    export_analytics(tracks, team_ball_control, settings)
    build_pdf_report_from_notebook(settings)