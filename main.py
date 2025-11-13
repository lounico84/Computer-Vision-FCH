import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read Video
    #video_frames = read_video('input_videos_match/Test/wiesendangen_test_clip_short.mp4')
    video_frames = read_video('input_videos_match/Test/kuesnacht_test_clip2.MP4')

    # Initialize Tracker
    tracker =  Tracker('yolo_training/models/fifth_model/run1/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='project/Computer-Vision-FCH/stubs/track_stubs_k.pkl')
    '''
    # Save cropped image

    frame_idx = 1
    track_id = 17 

    referee = tracks['referees'][frame_idx][track_id]
    bbox = referee['bbox']
    frame = video_frames[frame_idx]

    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = frame[y1:y2, x1:x2]

    cv2.imwrite("output_video_match/cropped_player_19.jpg", cropped_image)
    exit()
    '''

    # Intepolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks)
    team_assigner.assign_referee_color(video_frames, tracks)
    team_assigner.save_color_debug("output_video_match/color_debug_k.png")
    
    # 1. PASS: nur Farben/Abstände messen und Votes sammeln
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team_assigner.get_player_team(
                video_frames[frame_number],
                track['bbox'],
                player_id
            )
    
    # Nach dem 1. PASS: Mehrheits-Team pro Track bestimmen
    for player_id, votes_dict in team_assigner.team_vote_counts.items():
        v1 = votes_dict.get(1, 0)
        v2 = votes_dict.get(2, 0)
        # einfache Mehrheit; bei Gleichstand Team 1
        majority_team = 1 if v1 >= v2 else 2
        team_assigner.player_team_dict[player_id] = majority_team

    # Finale Entscheidung pro Track: Ref oder Team 1/2
    final_labels = {}  # player_id -> 0,1,2

    min_votes = 7
    min_ratio = 0.8
    min_observations = 10

    for player_id, obs in team_assigner.player_obs_counts.items():
        votes = team_assigner.ref_vote_counts.get(player_id, 0)

        # Default-Team jetzt: Mehrheits-Team aus oben
        default_team = team_assigner.player_team_dict.get(player_id, 1)

        if (
            team_assigner.referee_color is not None
            and obs >= min_observations
            and votes >= min_votes
            and votes >= min_ratio * obs
        ):
            final_labels[player_id] = 0   # Ref
        else:
            final_labels[player_id] = default_team

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

    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)
    
    # Draw Output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save Video
    save_video(output_video_frames, 'output_video_match/output_video_k.avi')

if __name__ == '__main__':
    main()