import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.ball_detection import BallDetection
from src.detection_overlay_renderer import DetectionOverlayRenderer

# Az asztal posztójának HSV tartománya (a videóhoz hangolandó!)
CLOTH_LOWER = np.array([40, 80, 100])
CLOTH_UPPER = np.array([80, 255, 255])

# Golyószínek: név -> [(alsó HSV, felső HSV), ...] (szintén hangolandó)
COLOR_RANGES = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    'yellow': [((20, 100, 100), (30, 255, 255))],
    'brown':  [((8, 100, 40), (20, 220, 140))],
    'green':  [((65, 60, 20), (88, 255, 120))],
    'blue':   [((90, 80, 50), (135, 255, 255))],
    'pink':   [((0, 40, 150), (10, 120, 255)), ((170, 40, 150), (180, 120, 255))],
    'black':  [((0, 0, 0), (180, 255, 50))],
    'white':  [((0, 0, 200), (180, 40, 255))],
}


class NeuralNetworkBallDetector:
    """YOLO-based snooker ball detector with cloth filtering and colour naming."""

    def __init__(
        self,
        model_path: str = "src/neural_network/v1.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        renderer: DetectionOverlayRenderer | None = None,
        filter_non_cloth: bool = True,
        min_cloth_ratio: float = 0.35,
        classify_colors: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model weights not found at: {model_path}")

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.renderer = renderer or DetectionOverlayRenderer()
        self.filter_non_cloth = filter_non_cloth
        self.min_cloth_ratio = min_cloth_ratio
        self.classify_colors = classify_colors
        self.last_detections: List[BallDetection] = []

    # ---------- segédfüggvények ----------

    @staticmethod
    def _surrounded_by_cloth(cloth_mask, bbox, min_ratio, pad=0.6) -> bool:
        """A golyó dobozát körülvevő téglalap keret hány százaléka poszto."""
        H, W = cloth_mask.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)
        if x2 <= x1 or y2 <= y1:
            return False

        px = int((x2 - x1) * pad)
        py = int((y2 - y1) * pad)
        ox1, oy1 = max(x1 - px, 0), max(y1 - py, 0)
        ox2, oy2 = min(x2 + px, W), min(y2 + py, H)

        outer = cloth_mask[oy1:oy2, ox1:ox2]
        ring = np.ones(outer.shape, np.uint8)
        ring[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = 0

        total = cv2.countNonZero(ring)
        if total == 0:
            return False
        hits = cv2.countNonZero(cv2.bitwise_and(outer, outer, mask=ring))
        return hits / total >= min_ratio

    @staticmethod
    def _classify_color(hsv, cloth_mask, bbox) -> Optional[str]:
        """A doboz közepén lévő, nem poszto pixelek szavazata alapján színnév."""
        H, W = cloth_mask.shape[:2]
        x1, y1, x2, y2 = bbox
        mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
        x1, y1 = max(x1 + mx, 0), max(y1 + my, 0)
        x2, y2 = min(x2 - mx, W), min(y2 - my, H)
        if x2 <= x1 or y2 <= y1:
            return None

        roi_hsv = hsv[y1:y2, x1:x2]
        valid = cv2.bitwise_not(cloth_mask[y1:y2, x1:x2])
        total = cv2.countNonZero(valid)
        if total < 4:
            return None

        best_name, best_ratio = None, 0.0
        for name, ranges in COLOR_RANGES.items():
            m = np.zeros(valid.shape, np.uint8)
            for lo, hi in ranges:
                m |= cv2.inRange(roi_hsv, np.array(lo), np.array(hi))
            ratio = cv2.countNonZero(cv2.bitwise_and(m, valid)) / total
            if ratio > best_ratio:
                best_name, best_ratio = name, ratio

        return best_name if best_ratio >= 0.25 else None

    # ---------- detektálás ----------

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
            self.last_detections = detections
            return detections

        result = results[0]
        names = result.names  # e.g., {0: 'ball'}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cloth_mask = cv2.inRange(hsv, CLOTH_LOWER, CLOTH_UPPER)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)

            # Hamis találatok (kéz, dákó, szegély): golyót mindig poszto vesz körül
            if self.filter_non_cloth and not self._surrounded_by_cloth(
                cloth_mask, bbox, self.min_cloth_ratio
            ):
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names.get(cls_id, "ball")

            if self.classify_colors:
                color_name = self._classify_color(hsv, cloth_mask, bbox)
                if color_name is not None:
                    label = color_name

            detections.append(
                BallDetection(
                    bbox=bbox,
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