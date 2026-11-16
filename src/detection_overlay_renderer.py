from typing import List, Tuple
import cv2
import numpy as np
from src.ball_detection import BallDetection


class DetectionOverlayRenderer:
    """Handles visual rendering of ball detections onto a transparent-equivalent black overlay."""

    def __init__(
        self,
        box_color: Tuple[int, int, int] = (0, 0, 255),      # Red in BGR
        center_color: Tuple[int, int, int] = (0, 255, 255), # Yellow in BGR
        text_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        font_scale: float = 0.9,
        text_thickness: int = 2,
        show_confidence: bool = False,
    ):
        self.box_color = box_color
        self.center_color = center_color
        self.text_color = text_color
        self.thickness = thickness
        self.font_scale = font_scale
        self.text_thickness = text_thickness
        self.show_confidence = show_confidence

    def render(self, frame_shape: Tuple[int, ...], detections: List[BallDetection]) -> np.ndarray:
        overlay = np.zeros(frame_shape, dtype=np.uint8)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center

            # Draw bounding box and center point
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.box_color, self.thickness)
            cv2.circle(overlay, (cx, cy), radius=3, color=self.center_color, thickness=-1)

            # Draw confidence label above the bounding box
            label_text = det.label
            if self.show_confidence:
                label_text += f" ({det.confidence:.2f})"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_thickness)
            text_x = min(max(x1, 0), frame_shape[1] - tw - 4)   # <- ez az új sor
            text_y = max(y1 - 8, th + 4)
            cv2.rectangle(overlay, 
                          (text_x, text_y - th - 4), 
                          (text_x + tw + 4, text_y + baseline),
                           (0, 0, 0),
                            -1,)
            cv2.putText(
                overlay,
                label_text,
                (text_x + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                self.text_thickness,
                cv2.LINE_AA,
            )

        return overlay