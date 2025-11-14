import cv2

# Read all frames from a video file into a list
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video or read error
            break
        frames.append(frame)
    return frames

# Save a list of frames as a video file
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Create video writer
    out = cv2.VideoWriter(output_video_path, fourcc, 60, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    # Write each frame to the output video
    for frame in output_video_frames:
        out.write(frame)

    # Close file
    out.release()