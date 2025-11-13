import cv2
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    # Read Video
    video_frames = read_video('input_videos_match/Test/kuesnacht_test_clip.MP4')

    # Initialize Tracker
    tracker =  Tracker('yolo_training/models/fifth_model/run1/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='project/Computer-Vision-FCH/stubs/track_stubs.pkl')

    # Save cropped image
    '''
    frame_idx = 1
    track_id = 17 

    player = tracks['players'][frame_idx][track_id]
    bbox = player['bbox']
    frame = video_frames[frame_idx]

    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = frame[y1:y2, x1:x2]

    cv2.imwrite("output_video_match/cropped_player_17.jpg", cropped_image)
    exit()
    '''

    # Assign Player Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks)
    team_assigner.assign_referee_color(video_frames, tracks)
    team_assigner.save_color_debug("output_video_match/color_debug.png")
    
    # 1. PASS: nur Farben/Abstände messen und Votes sammeln
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team_assigner.get_player_team(
                video_frames[frame_number],
                track['bbox'],
                player_id
            )

    # Finale Entscheidung pro Track: Ref oder Team 1/2
    final_labels = {}  # player_id -> 0,1,2

    # Schwellwerte für Ref-Entscheidung
    min_votes = 7            # wie oft der Track „ref-ähnlich“ war
    min_ratio = 0.8          # wie viele Frames ein Track existiert hat (votes / observations)
    min_observations = 10    # Track muss mindestens 10 Mal beobachtet worden sein

    for player_id, obs in team_assigner.player_obs_counts.items():
        votes = team_assigner.ref_vote_counts.get(player_id, 0)

        # Bedingung: mind. 7 Ref-Votes und zugleich in >= 80s% der beobachteten Frames Ref-ähnlich
        if (
            team_assigner.referee_color is not None
            and obs >= min_observations
            and votes >= min_votes
            and votes >= min_ratio * obs
        ):
            # Track wird als Schiri/Linienrichter interpretiert
            final_labels[player_id] = 0
        else:
            # Default-Team aus dem ersten Pass
            final_labels[player_id] = team_assigner.player_team_dict.get(player_id, 1)

    # 2. PASS: Tracks anhand finaler Labels umbauen
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in list(player_track.items()):
            team = final_labels.get(player_id, 1)

            if team == 0:
                # kompletter Track ist Schiri/Linienrichter
                tracks['referees'][frame_number][player_id] = track
                del tracks['players'][frame_number][player_id]
            else:
                tracks['players'][frame_number][player_id]['team'] = team
                tracks['players'][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw Output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_video_match/output_video.avi')

if __name__ == '__main__':
    main()