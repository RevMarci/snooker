import cv2
import numpy as np

class PerspectiveCorrector:
    def __init__(self, src_points, dst_width=1920, dst_height=1080):
        """
        src_points: A 4 sarokpont az eredeti videón (bal felső, jobb felső, bal alsó, jobb alsó)
        dst_width, dst_height: A kimeneti kép méretei. A snooker asztal aránya kb 2:1.
        """
        self.dst_width = dst_width
        self.dst_height = dst_height
        
        # Forrás pontok (eredeti videó)
        pts1 = np.float32(src_points)
        
        # Cél pontok (tökéletes felülnézeti téglalap)
        # Sorrend: bal-felső, jobb-felső, bal-alsó, jobb-alsó
        pts2 = np.float32([
            [0, 0], 
            [dst_width, 0], 
            [0, dst_height], 
            [dst_width, dst_height]
        ])
        
        # Transzformációs mátrix kiszámítása
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def process(self, frame):
        # Transzformáció alkalmazása a képkockára
        return cv2.warpPerspective(frame, self.matrix, (self.dst_width, self.dst_height))