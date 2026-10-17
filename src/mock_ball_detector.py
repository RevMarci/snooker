import cv2

class MockBallDetector:
    def __init__(self):
        self.x = 0
        self.y = 100
        self.box_size = 50
        self.speed = 5

    def process_frame(self, frame):
        output_frame = frame.copy()

        height, width, _ = output_frame.shape

        self.x += self.speed

        if self.x > width:
            self.x = -self.box_size

        top_left = (self.x, self.y)
        bottom_right = (self.x + self.box_size, self.y + self.box_size)
        
        cv2.rectangle(output_frame, top_left, bottom_right, (0, 0, 255), -1)

        text_pos = (self.x, self.y - 10)
        cv2.putText(output_frame, f"X: {self.x}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_frame
        