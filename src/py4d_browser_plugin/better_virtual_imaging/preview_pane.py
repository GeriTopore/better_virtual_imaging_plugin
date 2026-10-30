import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from py4D_browser.utils import DetectorShape, pg_point_roi

# The main window's real-space selector pens (see update_realspace_detector
# in update_views.py), so the cursors look the same here.
CURSOR_PEN = {"color": "g", "width": 6}
CURSOR_HANDLE_PEN = {"color": "r", "width": 9}
CURSOR_HOVER_PEN = {"color": "c", "width": 6}
CURSOR_HOVER_HANDLE_PEN = {"color": "c", "width": 9}

# Thinner than the main window's detector, as there can be many apertures.
APERTURE_PEN = {"color": "g", "width": 2}
APERTURE_HOVER_PEN = {"color": "c", "width": 2}


class PreviewPane(QGroupBox):
    """
    One diffraction pattern and virtual image pair, laid out like the main
    window: the pattern (with this pane's virtual apertures drawn over it) on
    the left, and the virtual image they make (with a scan-position cursor)
    on the right. The window does the computing; the pane holds the views and
    reports interaction through its signals.
    """

    # The real-space cursor moved (continuously for a point, on release for a
    # rectangle -- as in the main window).
    sig_scan_selection_changed = QtCore.pyqtSignal()
    # An aperture was moved, resized or removed.
    sig_apertures_changed = QtCore.pyqtSignal()
    # A left click on the diffraction pattern, outside any existing aperture,
    # at (qx, qy) in data coordinates.
    sig_dp_clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, title):
        super().__init__(title)

        self.apertures = []  # pg.CircleROIs, in the order they were added
        self.rs_selector = None
        self.rs_shape = None
        self.last_dp = None
        self._has_shown_dp = False

        self.dp_view = pg.ImageView()
        self.dp_view.setImage(np.zeros((512, 512)))
        self.dp_view.scene.sigMouseClicked.connect(self._on_dp_clicked)

        self.vi_view = pg.ImageView()
        self.vi_view.setImage(np.zeros((25, 25)))

        # pyqtgraph's own ROI tool would only be confused with the apertures
        # and cursor.
        for view in (self.dp_view, self.vi_view):
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()

        self.dp_label = QLabel()
        self.vi_label = QLabel()

        self.clear_apertures_button = QPushButton("Clear Apertures")
        self.clear_apertures_button.clicked.connect(self.clear_apertures)

        header = QHBoxLayout()
        header.addStretch()
        header.addWidget(self.clear_apertures_button)

        views = QGridLayout()
        views.addWidget(self.dp_view, 0, 0)
        views.addWidget(self.vi_view, 0, 1)
        views.addWidget(self.dp_label, 1, 0)
        views.addWidget(self.vi_label, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addLayout(views)
        self.setLayout(layout)

    def set_colormaps(self, realspace_cmap, diffraction_cmap):
        self.vi_view.setColorMap(realspace_cmap)
        self.dp_view.setColorMap(diffraction_cmap)

    def set_click_to_add(self, enabled):
        viewport = self.dp_view.ui.graphicsView.viewport()
        if enabled:
            viewport.setCursor(QtCore.Qt.CrossCursor)
        else:
            viewport.unsetCursor()

    # --- Diffraction pattern and apertures ---

    def show_dp(self, dp, scale_fn, relevel):
        self.last_dp = dp
        # Transposed to match the main window's diffraction_space_widget
        # convention (see _render_diffraction_image in update_views.py).
        scaled = scale_fn(dp).T

        if relevel or not self._has_shown_dp:
            levels = tuple(np.percentile(scaled, [0.1, 99.9]))
            self.dp_view.setImage(
                scaled,
                autoLevels=False,
                levels=levels,
                autoRange=not self._has_shown_dp,
            )
            self._has_shown_dp = True
        else:
            # Leave levels untouched, so any range the user set by hand on
            # the histogram survives the cursor moving.
            self.dp_view.setImage(scaled, autoLevels=False, autoRange=False)

    def _on_dp_clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        view = self.dp_view.getView()
        if not view.sceneBoundingRect().contains(ev.scenePos()):
            return

        # The pattern is displayed transposed, with data pixel (qx, qy)
        # covering [qy, qy + 1] x [qx, qx + 1] in view coordinates.
        pos = view.mapSceneToView(ev.scenePos())
        shown_y, shown_x = self.dp_view.image.shape[:2]
        if not (0 <= pos.x() < shown_y and 0 <= pos.y() < shown_x):
            return
        qx, qy = pos.y() - 0.5, pos.x() - 0.5

        # A click on an existing aperture is the user reaching for it, not
        # asking for a second one on top.
        for (aqx, aqy), radius in self.aperture_geometries():
            if np.hypot(qx - aqx, qy - aqy) < radius:
                return

        self.sig_dp_clicked.emit(qx, qy)

    def add_aperture(self, qx, qy, radius):
        # The inverse of _aperture_geometry: centred on data pixel (qx, qy),
        # which is displayed at (qy + 0.5, qx + 0.5).
        roi = pg.CircleROI(
            (qy + 0.5 - radius, qx + 0.5 - radius),
            (2 * radius, 2 * radius),
            pen=APERTURE_PEN,
            hoverPen=APERTURE_HOVER_PEN,
            removable=True,  # right-click > Remove ROI
        )
        roi.sigRegionChangeFinished.connect(
            lambda *_: self.sig_apertures_changed.emit()
        )
        roi.sigRemoveRequested.connect(self.remove_aperture)
        self.dp_view.getView().addItem(roi)
        self.apertures.append(roi)

    def remove_aperture(self, roi):
        self.dp_view.getView().removeItem(roi)
        self.apertures.remove(roi)
        self.sig_apertures_changed.emit()

    def clear_apertures(self):
        if not self.apertures:
            return
        for roi in self.apertures:
            self.dp_view.getView().removeItem(roi)
        self.apertures = []
        self.sig_apertures_changed.emit()

    @staticmethod
    def _aperture_geometry(roi):
        radius = roi.size()[0] / 2
        qy = roi.pos()[0] + radius - 0.5
        qx = roi.pos()[1] + radius - 0.5
        return (qx, qy), radius

    def aperture_geometries(self):
        """Each aperture as ((qx, qy), radius), in data coordinates."""
        return [self._aperture_geometry(roi) for roi in self.apertures]

    # --- Virtual image and real-space cursor ---

    def show_vi(self, vimg):
        # Transposed to match the main window's real_space_widget, which the
        # cursor reading below relies on. Zoom is only reset when the shape
        # changes, so updates don't undo the user's panning/zooming.
        self.vi_view.setImage(
            vimg.T,
            autoLevels=True,
            autoRange=self.vi_view.image.shape != vimg.T.shape,
        )

    def _remove_selector(self):
        if self.rs_selector is not None:
            self.vi_view.getView().removeItem(self.rs_selector)
            self.rs_selector = None

    def set_point_selector(self, xc, yc):
        self._remove_selector()
        self.rs_selector = pg_point_roi(
            self.vi_view.getView(),
            center=(yc - 0.5, xc - 0.5),
            pen=CURSOR_PEN,
            hoverPen=CURSOR_HOVER_PEN,
        )
        self.rs_selector.sigRegionChanged.connect(
            lambda *_: self.sig_scan_selection_changed.emit()
        )
        self.rs_shape = DetectorShape.POINT

    def set_rect_selector(self, slice_x, slice_y):
        self._remove_selector()
        self.rs_selector = pg.RectROI(
            [slice_y.start, slice_x.start],
            [slice_y.stop - slice_y.start, slice_x.stop - slice_x.start],
            pen=CURSOR_PEN,
            handlePen=CURSOR_HANDLE_PEN,
            hoverPen=CURSOR_HOVER_PEN,
            handleHoverPen=CURSOR_HOVER_HANDLE_PEN,
            # Unlike the main window's, snapped to whole scan pixels, so the
            # rectangle drawn is exactly the one averaged over.
            translateSnap=True,
            scaleSnap=True,
        )
        self.vi_view.getView().addItem(self.rs_selector)
        self.rs_selector.sigRegionChangeFinished.connect(
            lambda *_: self.sig_scan_selection_changed.emit()
        )
        self.rs_shape = DetectorShape.RECTANGULAR

    def get_scan_selection(self, Rshape):
        """
        The cursor's selection in data coordinates, as the main window's
        get_virtual_image_detector gives its own: an (xc, yc) point, or
        [slice_x, slice_y] for a rectangle (empty if it's off the scan).
        """
        if self.rs_shape is DetectorShape.POINT:
            y0, x0 = self.rs_selector.saveState()["pos"]
            xc, yc = int(x0 + 1), int(y0 + 1)
            xc = int(np.clip(xc, 0, Rshape[0] - 1))
            yc = int(np.clip(yc, 0, Rshape[1] - 1))
            return xc, yc

        # Read from the (snapped) position and size rather than with
        # getArraySlice as the main window does, which rounds float noise in
        # the view transform outwards and can take in an extra row/column.
        y0, x0 = self.rs_selector.pos()
        ny, nx = self.rs_selector.size()

        def clipped(start, stop, n):
            return slice(
                int(np.clip(round(start), 0, n)), int(np.clip(round(stop), 0, n))
            )

        return [clipped(x0, x0 + nx, Rshape[0]), clipped(y0, y0 + ny, Rshape[1])]

    def scan_centre(self, Rshape):
        """The scan position at the cursor, or the centre of its rectangle."""
        if self.rs_shape is DetectorShape.POINT:
            return self.get_scan_selection(Rshape)
        slice_x, slice_y = self.get_scan_selection(Rshape)
        if slice_x.start >= slice_x.stop or slice_y.start >= slice_y.stop:
            return Rshape[0] // 2, Rshape[1] // 2
        return (
            (slice_x.start + slice_x.stop - 1) // 2,
            (slice_y.start + slice_y.stop - 1) // 2,
        )
