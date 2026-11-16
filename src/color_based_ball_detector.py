import cv2
import numpy as np


class ColorBasedBallDetector:
    def __init__(self):
        # Az asztal posztójának HSV tartománya (a konkrét videóhoz hangolandó)
        self.cloth_lower = np.array([55, 130, 30])
        self.cloth_upper = np.array([65, 230, 190])

        # Golyószínek: név -> [(alsó, felső), ...]. Minden érték hangolandó!
        # A piros / rózsaszín / barna átfedhet, ezért a golyó pixeleinek
        # többségi szavazata dönt, nem egyetlen küszöb.
        self.color_ranges = {
            'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
            'yellow': [((20, 100, 100), (30, 255, 255))],
            'brown':  [((8, 100, 40), (20, 220, 140))],
            'green':  [((70, 100, 40), (90, 255, 200))],
            'blue':   [((100, 150, 50), (140, 255, 255))],
            'pink':   [((0, 40, 150), (10, 120, 255)), ((170, 40, 150), (180, 120, 255))],
            'black':  [((0, 0, 0), (180, 255, 50))],
            'white':  [((0, 0, 200), (180, 40, 255))],
        }

        self.bgr_colors = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'brown': (19, 69, 139),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'pink': (203, 192, 255),
            'black': (50, 50, 50),
            'white': (255, 255, 255),
        }

        # Golyó méret- és alakszűrők (felbontásfüggő, hangolandó)
        self.min_area = 30
        self.max_area = 1500
        self.min_circularity = 0.6
        self.min_vote_ratio = 0.3

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Az utolsó képkocka találatai: [(név, (x, y), sugár), ...]
        self.detections = []

    def _color_masks(self, hsv):
        masks = {}
        for name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            masks[name] = mask
        return masks

    def process_frame(self, frame):
        overlay = np.zeros_like(frame)
        self.detections = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1) Poszto kiválasztása és a játéktér meghatározása
        cloth = cv2.inRange(hsv, self.cloth_lower, self.cloth_upper)
        cloth = cv2.morphologyEx(cloth, cv2.MORPH_OPEN, self.kernel)
        cloth = cv2.morphologyEx(cloth, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return overlay

        table = max(contours, key=cv2.contourArea)
        field = np.zeros_like(cloth)
        cv2.drawContours(field, [table], -1, 255, -1)

        # 2) Golyójelöltek: ami az asztalon belül van, de nem poszto
        candidates = cv2.bitwise_and(cv2.bitwise_not(cloth), field)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, self.kernel)

        masks = self._color_masks(hsv)

        contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 2:
                continue

            circularity = area / (np.pi * radius * radius)
            if circularity < self.min_circularity:
                continue

            # 3) Színosztályozás a golyó pixeleinek szavazata alapján
            x0, y0, w, h = cv2.boundingRect(cnt)
            blob = np.zeros((h, w), np.uint8)
            cv2.drawContours(blob, [cnt - np.array([x0, y0])], -1, 255, -1)
            blob = cv2.erode(blob, self.kernel)  # a szélső, kevert színű pixelek kiesnek
            total = cv2.countNonZero(blob)
            if total == 0:
                continue

            best_name, best_ratio = None, 0.0
            for name, mask in masks.items():
                roi = mask[y0:y0 + h, x0:x0 + w]
                ratio = cv2.countNonZero(cv2.bitwise_and(roi, blob)) / total
                if ratio > best_ratio:
                    best_name, best_ratio = name, ratio

            if best_name is None or best_ratio < self.min_vote_ratio:
                continue

            center = (int(cx), int(cy))
            r = int(radius)
            self.detections.append((best_name, center, r))

            color = self.bgr_colors[best_name]
            cv2.circle(overlay, center, r, color, 2)
            cv2.putText(overlay, best_name, (center[0] - 25, center[1] - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return overlay