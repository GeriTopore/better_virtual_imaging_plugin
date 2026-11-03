import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from py4DSTEM.process.diffraction.digital_dark_field import aperture_array_generator

ORIGIN_PEN = {"color": "w", "width": 2}
G1_PEN = {"color": (255, 140, 0), "width": 2}
G2_PEN = {"color": (0, 200, 255), "width": 2}
HOVER_PEN = {"color": "c", "width": 2}
RING_COLOR = (255, 255, 0)
RING_PEN = {"color": RING_COLOR, "width": 3}

# How opaque every aperture's fill is (0-1), here and in saved figures.
APERTURE_FILL_ALPHA = 0.2


def fill_brush(color):
    """A translucent brush for filling an aperture, from an RGB (0-255) colour."""
    return pg.mkBrush(*color, int(round(255 * APERTURE_FILL_ALPHA)))


def _to_view(q):
    # Diffraction patterns are displayed transposed, with data pixel (qx, qy)
    # centred at (qy + 0.5, qx + 0.5) in view coordinates.
    return pg.Point(q[1] + 0.5, q[0] + 0.5)


def _to_data(point):
    return np.array([point.y() - 0.5, point.x() - 0.5])


class GVectorArray(QtCore.QObject):
    """
    An array of circular apertures on the lattice spanned by two g-vectors,
    drawn on a diffraction pattern view: draggable markers for the origin and
    the tips of g1 and g2 (moving the origin carries the vectors with it),
    and a ring for each aperture. The array extends `n1` and `n2` steps
    either side of the origin along g1 and g2, within the pattern, shifted
    by `s1` g1 + `s2` g2 (e.g. 0.5 for half-order spots).

    Given `peaks_fn` (returning an (N, 2) array of Bragg peak (qx, qy), or
    None), a marker dropped near a peak snaps onto it while `snap` is on.
    """

    # The array changed: continuously while a marker is dragged...
    sig_changed = QtCore.pyqtSignal()
    # ...and once when it's dropped, or when anything else changes.
    sig_change_finished = QtCore.pyqtSignal()

    def __init__(self, view_box, Qshape, origin, g1, g2, radius, peaks_fn=None):
        super().__init__()

        self.Qshape = Qshape
        self.origin = np.asarray(origin, dtype=float)
        self.g1 = np.asarray(g1, dtype=float)
        self.g2 = np.asarray(g2, dtype=float)
        self.radius = radius
        self.n1 = 3
        self.n2 = 3
        self.s1 = 0.0
        self.s2 = 0.0
        self.include_origin = False
        self.peaks_fn = peaks_fn
        self.snap = True

        self.rings = pg.ScatterPlotItem(pxMode=False)  # sizes in data pixels
        self.g1_line = pg.PlotCurveItem(pen=G1_PEN)
        self.g2_line = pg.PlotCurveItem(pen=G2_PEN)

        def target(pen, label):
            return pg.TargetItem(
                size=12,
                symbol="o",
                pen=pen,
                hoverPen=HOVER_PEN,
                brush=None,
                label=label,
                labelOpts={"color": pen["color"], "offset": (8, -8)},
            )

        self.origin_target = target(ORIGIN_PEN, "O")
        self.g1_target = target(G1_PEN, "g1")
        self.g2_target = target(G2_PEN, "g2")

        self.items = [
            self.rings,
            self.g1_line,
            self.g2_line,
            self.origin_target,
            self.g1_target,
            self.g2_target,
        ]
        for z, item in enumerate(self.items):
            item.setZValue(10 + z)  # above the image and other apertures
            view_box.addItem(item)
        self.view_box = view_box

        self._place_targets()
        self._redraw()

        self.origin_target.sigPositionChanged.connect(self._on_origin_moved)
        self.g1_target.sigPositionChanged.connect(self._on_tip_moved)
        self.g2_target.sigPositionChanged.connect(self._on_tip_moved)
        for item in (self.origin_target, self.g1_target, self.g2_target):
            item.sigPositionChangeFinished.connect(self._on_marker_dropped)

    def _place_targets(self):
        # Without re-entering _on_origin_moved/_on_tip_moved.
        for item, q in (
            (self.origin_target, self.origin),
            (self.g1_target, self.origin + self.g1),
            (self.g2_target, self.origin + self.g2),
        ):
            item.blockSignals(True)
            item.setPos(_to_view(q))
            item.blockSignals(False)

    def _on_origin_moved(self, *_):
        self.origin = _to_data(self.origin_target.pos())
        self._place_targets()
        self._redraw()
        self.sig_changed.emit()

    def _on_tip_moved(self, *_):
        self.g1 = _to_data(self.g1_target.pos()) - self.origin
        self.g2 = _to_data(self.g2_target.pos()) - self.origin
        self._redraw()
        self.sig_changed.emit()

    def snap_distance(self):
        # How near a peak a marker must be dropped to snap onto it.
        return max(2 * self.radius, 3.0)

    def snap_all(self, emit=True):
        """
        Snap every marker onto the peak nearest to it, however far: O, then
        g1's tip (to a peak other than O's), then g2's (to one not in line
        with g1, so the lattice can't collapse). Returns whether there were
        peaks to snap to.
        """
        peaks = self.peaks_fn() if self.peaks_fn is not None else None
        if peaks is None or len(peaks) == 0:
            return False

        def nearest(candidates, q):
            return candidates[np.argmin(np.hypot(*(candidates - q).T))]

        origin = nearest(peaks, self.origin)
        vectors = peaks - origin
        lengths = np.hypot(*vectors.T)
        others = lengths > 1
        if others.any():
            g1 = nearest(vectors[others], self.g1)
            # Not within ~15 degrees of g1, either way.
            cross = vectors[:, 0] * g1[1] - vectors[:, 1] * g1[0]
            sines = np.abs(cross) / np.maximum(lengths * np.hypot(*g1), 1e-12)
            off_line = others & (sines > np.sin(np.radians(15)))
            g2 = nearest(vectors[off_line], self.g2) if off_line.any() else self.g2
            self.g1, self.g2 = g1, g2
        self.origin = origin

        self._place_targets()
        self._redraw()
        if emit:
            self.sig_change_finished.emit()
        return True

    def _on_marker_dropped(self, item):
        peaks = self.peaks_fn() if self.snap and self.peaks_fn is not None else None
        if peaks is not None and len(peaks):
            distances = np.hypot(*(peaks - _to_data(item.pos())).T)
            nearest = np.argmin(distances)
            if distances[nearest] <= self.snap_distance():
                # Updates the origin or vector as a drag would.
                item.setPos(_to_view(peaks[nearest]))
        self.sig_change_finished.emit()

    def _redraw(self):
        for line, g in ((self.g1_line, self.g1), (self.g2_line, self.g2)):
            start, end = _to_view(self.origin), _to_view(self.origin + g)
            line.setData([start.x(), end.x()], [start.y(), end.y()])
        self.rings.setData(
            [
                {
                    "pos": _to_view(q),
                    "size": 2 * self.radius,
                    "pen": pg.mkPen(**RING_PEN),
                    "brush": fill_brush(RING_COLOR),
                    "symbol": "o",
                }
                for q in self.positions()
            ]
        )

    def positions(self):
        """The aperture centres, as an (N, 2) array of (qx, qy)."""
        if np.hypot(*self.g1) < 1 or np.hypot(*self.g2) < 1:
            return np.zeros((0, 2))
        positions, _ = aperture_array_generator(
            self.Qshape,
            self.origin,
            0,  # pad: keep everything on the pattern
            "array",
            self.g1,
            self.g2,
            s1=self.s1,
            s2=self.s2,
            n1lims=(-self.n1, self.n1),
            n2lims=(-self.n2, self.n2),
            # The origin is the only lattice point that can be at radius 0.
            r1=0 if self.include_origin else 1e-6,
            r2=np.inf,
            returns="both",
        )
        return positions.reshape(-1, 2)

    def apertures(self):
        """Each aperture as ((qx, qy), radius), in data coordinates."""
        return [((qx, qy), self.radius) for qx, qy in self.positions()]

    def set_params(self, **params):
        """Set any of n1, n2, s1, s2, include_origin, radius and snap."""
        for name, value in params.items():
            if name not in ("n1", "n2", "s1", "s2", "include_origin", "radius", "snap"):
                raise TypeError(f"Unknown parameter {name}")
            setattr(self, name, value)
        self._redraw()
        if set(params) != {"snap"}:  # snapping alone changes no apertures
            self.sig_change_finished.emit()

    def set_visible(self, visible):
        for item in self.items:
            item.setVisible(visible)

    def marker_at(self, scene_pos):
        """Whether a (visible) marker is under a point in scene coordinates."""
        return any(
            item.isVisible() and item.sceneBoundingRect().contains(scene_pos)
            for item in (self.origin_target, self.g1_target, self.g2_target)
        )
