import cv2
import numpy as np

# --- Beállítások -----------------------------------------------------------
# A kimenet álló (portrait) kép, mint a mintaképen: az asztal hosszabb oldala
# függőleges. A zöld felület aránya a mintakép alapján
VIEW_WIDTH = 480                     # a teljes kimenet szélessége (margóval együtt)
CLOTH_ASPECT = 1920 / 1080            # a zöld felület magasság / szélesség aránya

# Margó a zöld felület körül, a kimenet szélességének arányában. Itt látszik
# a banda és a 6 zseb. Nagyobb szám = több látszik az asztal széléből.
VIEW_MARGIN = 0.06

_margin = int(round(VIEW_WIDTH * VIEW_MARGIN))
_cloth_w = VIEW_WIDTH - 2 * _margin
_cloth_h = int(round(_cloth_w * CLOTH_ASPECT))
VIEW_HEIGHT = _cloth_h + 2 * _margin

# Forgatás órajárás szerint, fokban: 0, 90, 180 vagy 270.
#   0:   a kimenet úgy áll, ahogy a videón látszik (a hosszú oldal függőleges).
#   180: a mintakép tájolása: a baulk-vonal (a "D") alul van. A clip1-ben a
#        kamera a baulk-vonallal szembeni végnél áll, ezért kell 180.
#   90 / 270: ha az asztal hosszú oldala a videón vízszintes.
ROTATION = 180

# Az asztal zöld színtartománya HSV-ben (OpenCV: H 0-179). Ha a videó színei
# eltérnek, ezt kell hangolni.
HSV_LOWER = (35, 60, 40)
HSV_UPPER = (90, 255, 255)
# ---------------------------------------------------------------------------


class StableTableDetector:
    """Megkeresi a snooker asztalt a zöld posztó színe alapján, és visszaad
    4 sarokpontot (bal-felső, jobb-felső, bal-alsó, jobb-alsó sorrendben).

    Álló kamerát feltételez. A takarás (játékos, dákó) ellen az eddigi összes
    zöld maszk UNIÓJÁBÓL számol: amit egyszer már zöldnek láttunk, az az
    asztalhoz tartozik, akkor is, ha épp valaki eltakarja. A sarkok csak
    akkor változnak, ha az unió érdemben nőtt, így a kép nem mozog."""

    def __init__(self, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
                 min_area_ratio=0.05, work_scale=0.5, growth_tolerance=0.002):
        """
        min_area_ratio:   a kép hány %-át kell minimum kitöltenie a zöld
                          foltnak, hogy asztalnak fogadjuk el.
        work_scale:       a detektálás ekkora léptékű képen fut (gyorsabb).
        growth_tolerance: ennyivel (relatíve) kell nőnie a zöld uniónak, hogy
                          a sarkokat újraszámoljuk.
        """
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area_ratio = min_area_ratio
        self.work_scale = work_scale
        self.growth_tolerance = growth_tolerance
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        self.corners = None      # utolsó eredmény
        self.union = None        # az eddigi összes zöld maszk uniója
        self.union_area = 0      # az unió területe a legutóbbi újraszámításkor
        self.last_mask = None    # debughoz

    def detect(self, frame):
        """4x2 float32 sarokpontok (None, ha még nem találtunk asztalt)."""
        mask = self._green_mask(frame)
        self.last_mask = mask

        self.union = mask.copy() if self.union is None else cv2.bitwise_or(self.union, mask)
        area = cv2.countNonZero(self.union)

        if self.corners is not None and area <= self.union_area * (1 + self.growth_tolerance):
            return self.corners  # nincs érdemi változás -> a kép nem mozog

        quad = self._quad_from_mask(self.union)
        if quad is None:
            return self.corners

        self.corners = self._order_points(quad / self.work_scale)
        self.union_area = area
        return self.corners

    # ------------------------------------------------------------------
    def _green_mask(self, frame):
        small = cv2.resize(frame, None, fx=self.work_scale, fy=self.work_scale)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # lyukak (golyók, pöckök) betömése, zaj eltávolítása
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return mask

    def _quad_from_mask(self, mask):
        """A maszk legnagyobb foltjára négyszöget illeszt (work_scale léptékben)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_area_ratio * mask.shape[0] * mask.shape[1]:
            return None

        return self._fit_quad(cv2.convexHull(biggest))

    @staticmethod
    def _fit_quad(hull):
        """A kontúrra négyszöget illeszt (trapéz-szerű, nem szabályos)."""
        peri = cv2.arcLength(hull, True)
        for eps in np.linspace(0.01, 0.10, 19):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
            if len(approx) < 4:
                break
        # tartalék: legkisebb befoglaló forgatott téglalap
        return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)

    @staticmethod
    def _order_points(pts):
        """Sorrend: bal-felső, jobb-felső, bal-alsó, jobb-alsó."""
        s = pts.sum(axis=1)             # x + y
        d = pts[:, 0] - pts[:, 1]       # x - y
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmax(d)]
        bl = pts[np.argmin(d)]
        return np.float32([tl, tr, bl, br])


class TableTopView:
    """Megkeresi az asztalt a zöld alapján, és felülnézeti, álló képet ad."""

    def __init__(self):
        self.detector = StableTableDetector()
        # A zöld felület négyszöge a kimenet közepére kerül, körülötte a margó:
        # a warp a margóba a videó valódi képpontjait húzza (banda, zsebek).
        m = _margin
        self.dst_points = np.float32([
            [m, m], [m + _cloth_w, m],
            [m, m + _cloth_h], [m + _cloth_w, m + _cloth_h],
        ])

    def process(self, frame):
        """Felülnézeti kép; ha még nincs asztal, üres (fekete) kép."""
        corners = self.detector.detect(frame)
        if corners is None:
            return np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)

        tl, tr, bl, br = corners
        # órajárás szerinti sorrend, majd ROTATION/90 lépéssel körbeforgatva
        cw = [tl, tr, br, bl]
        k = (ROTATION // 90) % 4
        cw = cw[-k:] + cw[:-k] if k else cw
        src = [cw[0], cw[1], cw[3], cw[2]]  # bal-felső, jobb-felső, bal-alsó, jobb-alsó
        matrix = cv2.getPerspectiveTransform(np.float32(src), self.dst_points)
        return cv2.warpPerspective(frame, matrix, (VIEW_WIDTH, VIEW_HEIGHT))


_view = None


def table_top_view(frame):
    """Egyszerű belépési pont a main.py-nak: a TableTopView példányt maga
    hozza létre és tartja meg a képkockák között."""
    global _view
    if _view is None:
        _view = TableTopView()
    return _view.process(frame)