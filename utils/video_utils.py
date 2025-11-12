import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # Fallback

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps

def _fourcc_for_path(output_path: str):
    ext = os.path.splitext(output_path)[1].lower()
    # fallback
    if ext == ".avi":
        return cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter_fourcc(*'mp4v')

def save_video(output_video_frames, output_video_path, fps):
    if not output_video_frames:
        raise ValueError("No frames to write.")

    h, w = output_video_frames[0].shape[:2]
    fourcc = _fourcc_for_path(output_video_path)
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (w, h))
    for frame in output_video_frames:
        out.write(frame)
    out.release()