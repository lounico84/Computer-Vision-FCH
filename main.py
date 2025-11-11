from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read Video
    video_frames = read_video('input_videos_match/Test/wiesendangen_test_clip_short.mp4')

    # Initialize Tracker
    tracker =  Tracker('yolo_training/models/fifth_model/run1/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='project/Computer-Vision-FCH/stubs/track_stubs.pkl')

    # Draw Output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_video_match/output_video.avi')

if __name__ == '__main__':
    main()