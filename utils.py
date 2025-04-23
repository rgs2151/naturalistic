import cv2
import numpy as np

def load_video_frames(path: str) -> np.ndarray:
    """
    Load a video file into a NumPy array of frames.

    Parameters
    ----------
    path : str
        Path to the video file (e.g. 'video.mp4').

    Returns
    -------
    frames : np.ndarray
        Array of shape (num_frames, height, width, 3), dtype=uint8,
        with pixel values in RGB order.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames read from video: {path}")

    return np.stack(frames, axis=0)
