import cv2
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    # Read Video
    video_frames = read_video('input_videos_match/Test/wiesendangen_test_clip_short.mp4')

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
    
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_number],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_number][player_id]['team'] = team
            tracks['players'][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]


    # Draw Output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_video_match/output_video.avi')

if __name__ == '__main__':
    main()