from dataclasses import dataclass
from typing import Tuple


@dataclass
class BallDetection:
    """Data Transfer Object representing a single detected snooker ball."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    label: str
    confidence: float = 1.0
    class_id: int = 0