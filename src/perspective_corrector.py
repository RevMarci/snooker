import cv2
import numpy as np

_src_points = [
    [500, 50],   # Bal felső
    [1500, 50],  # Jobb felső
    [100, 900],  # Bal alsó
    [1800, 900]  # Jobb alsó
]

class PerspectiveCorrector:
    def __init__(self, src_points=_src_points, dst_width=1920, dst_height=1080):
        """
        src_points: A 4 sarokpont az eredeti videón (bal felső, jobb felső, bal alsó, jobb alsó)
        dst_width, dst_height: A kimeneti kép méretei. A snooker asztal aránya kb 2:1.
        """
        self.dst_width = dst_width
        self.dst_height = dst_height
        
        pts1 = np.float32(src_points)
        
        pts2 = np.float32([
            [0, 0], 
            [dst_width, 0], 
            [0, dst_height], 
            [dst_width, dst_height]
        ])
        
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def process(self, frame):
        return cv2.warpPerspective(frame, self.matrix, (self.dst_width, self.dst_height))
