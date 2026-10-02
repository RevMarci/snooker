import cv2

class VideoHandler:
    """Video handler class."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Error could not open video: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self):
        """Alows the use of the 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically closes the video at the end of the 'with' block."""
        self.close()

    def frames(self):
        """Generator, which feeds the video frame by frame."""
        # Ensure we start from the beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def close(self):
        """Frees up the resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            