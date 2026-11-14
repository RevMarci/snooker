import os
from typing import List
import numpy as np
from ultralytics import YOLO

from src.ball_detection import BallDetection
from src.detection_overlay_renderer import DetectionOverlayRenderer


class NeuralNetworkBallDetector:
    """YOLO-based snooker ball detector."""

    def __init__(
        self,
        model_path: str = "src/neural_network/v1.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        renderer: DetectionOverlayRenderer | None = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model weights not found at: {model_path}")

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.renderer = renderer or DetectionOverlayRenderer()
        self.last_detections: List[BallDetection] = []

    def detect(self, frame: np.ndarray) -> List[BallDetection]:
        """Runs inference on the frame and returns structured ball detections."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[BallDetection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names  # e.g., {0: 'ball'}

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names.get(cls_id, "ball")

            detections.append(
                BallDetection(
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    confidence=conf,
                    class_id=cls_id,
                    label=label,
                )
            )

        self.last_detections = detections
        return detections

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detects balls and returns a BGR overlay matching the pipeline contract."""
        detections = self.detect(frame)
        return self.renderer.render(frame.shape, detections)