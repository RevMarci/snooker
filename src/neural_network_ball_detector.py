import cv2
import numpy as np

class NeuralNetworkBallDetector:
    def __init__(self):
        self.x = 0
        self.y = 100
        self.box_size = 50
        self.speed = 5
        self.color = (0, 0, 255)

    def process_frame(self, frame):
        overlay = np.zeros_like(frame)

        height, width, _ = frame.shape
        self.x += self.speed

        if self.x > width:
            self.x = -self.box_size

        top_left = (self.x, self.y)
        bottom_right = (self.x + self.box_size, self.y + self.box_size)
        
        cv2.rectangle(overlay, top_left, bottom_right, self.color, -1)

        text_pos = (self.x, self.y - 10)
        cv2.putText(overlay, f"X: {self.x}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return overlay
