import cv2
import numpy as np

# --- Settings --------------------------------------------------------------
VIEW_WIDTH = 480
CLOTH_ASPECT = 1920 / 1080
VIEW_MARGIN = 0.06
ROTATION = 180

HSV_LOWER = (35, 60, 40)
HSV_UPPER = (90, 255, 255)
# ---------------------------------------------------------------------------

_margin = int(round(VIEW_WIDTH * VIEW_MARGIN))
_cloth_w = VIEW_WIDTH - 2 * _margin
_cloth_h = int(round(_cloth_w * CLOTH_ASPECT))
VIEW_HEIGHT = _cloth_h + 2 * _margin

class TableTopView:
    def __init__(self):
        self.transform_matrix = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        
        m = _margin
        self.dst_points = np.float32([
            [m, m], 
            [m + _cloth_w, m],
            [m, m + _cloth_h], 
            [m + _cloth_w, m + _cloth_h]
        ])

    def _get_green_mask(self, frame):
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        return mask

    def _find_exact_corners(self, mask):
        # CHAIN_APPROX_NONE needed, to get all contour points for line fitting
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < (mask.shape[0] * mask.shape[1] * 0.1):
            return None

        pts = cnt.reshape(-1, 2)

        # 1. Finding rough corner point indices within the contour
        s = pts.sum(axis=1)
        d = pts[:, 0] - pts[:, 1]
        
        tl_idx = np.argmin(s)
        br_idx = np.argmax(s)
        tr_idx = np.argmax(d)
        bl_idx = np.argmin(d)

        # Sorting the indices to get the 4 edges (bands)
        indices = sorted(list(set([tl_idx, tr_idx, br_idx, bl_idx])))
        
        if len(indices) != 4:
            return cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)

        lines = []
        for i in range(4):
            start_idx = indices[i]
            end_idx = indices[(i + 1) % 4]

            # Extracting points around the specified edge of the contour
            if start_idx < end_idx:
                segment = pts[start_idx:end_idx]
            else:
                segment = np.concatenate((pts[start_idx:], pts[:end_idx]))

            l = len(segment)
            trim = int(l * 0.15) # Trimming the ends to remove pockets (15-15%)
            
            if trim > 0 and (l - 2*trim) > 10:
                segment = segment[trim:-trim]

            # Fitting a line to the clean edge segment: [vx, vy, x0, y0]
            line = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line[0][0], line[1][0], line[2][0], line[3][0]
            lines.append((np.array([vx, vy]), np.array([x0, y0])))

        # 2. Calculating the intersection points of the 4 fitted lines
        exact_corners = []
        for i in range(4):
            v1, p1 = lines[i]
            v2, p2 = lines[(i + 1) % 4]

            # Linear system of equations for the intersection point of the two lines
            A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
            b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            try:
                t = np.linalg.solve(A, b)
                exact_corners.append(p1 + t[0] * v1)
            except np.linalg.LinAlgError:
                return cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)

        return np.float32(exact_corners)

    def _order_points(self, pts):
        s = pts.sum(axis=1)
        d = pts[:, 0] - pts[:, 1]
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmax(d)]
        bl = pts[np.argmin(d)]
        return np.float32([tl, tr, bl, br])

    def process(self, frame):
        if self.transform_matrix is None:
            mask = self._get_green_mask(frame)
            corners = self._find_exact_corners(mask)
            
            if corners is not None:
                ordered_corners = self._order_points(corners)
                tl, tr, bl, br = ordered_corners
                
                cw = [tl, tr, br, bl]
                k = (ROTATION // 90) % 4
                if k:
                    cw = cw[-k:] + cw[:-k]
                src_pts = np.float32([cw[0], cw[1], cw[3], cw[2]])
                
                self.transform_matrix = cv2.getPerspectiveTransform(src_pts, self.dst_points)
            else:
                return np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)

        return cv2.warpPerspective(frame, self.transform_matrix, (VIEW_WIDTH, VIEW_HEIGHT))


_view = None

def table_top_view(frame):
    global _view
    if _view is None:
        _view = TableTopView()
    return _view.process(frame)