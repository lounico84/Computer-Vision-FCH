from utils import read_video, save_video

def main():
    # Read Video
    video_frames = read_video('input_videos_match/Test/wiesendangen_test_clip_short.mp4')

    # Save Video
    save_video(video_frames, 'output_video_match/output_video.avi')

if __name__ == '__main__':
    main()