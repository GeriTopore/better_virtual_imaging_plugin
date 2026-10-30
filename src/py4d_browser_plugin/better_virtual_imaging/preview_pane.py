from functools import partial

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.utils import DetectorShape, pg_point_roi

from .lattice import G1_PEN, G2_PEN

# The main window's real-space selector pens (see update_realspace_detector
# in update_views.py), so the cursors look the same here.
CURSOR_PEN = {"color": "g", "width": 6}
CURSOR_HANDLE_PEN = {"color": "r", "width": 9}
CURSOR_HOVER_PEN = {"color": "c", "width": 6}
CURSOR_HOVER_HANDLE_PEN = {"color": "c", "width": 9}

# Thinner than the main window's detector, as there can be many apertures.
APERTURE_PEN = {"color": "g", "width": 2}
APERTURE_HOVER_PEN = {"color": "c", "width": 2}


def show_diffraction_image(view, image, scale_fn, relevel, first):
    # Transposed to match the main window's diffraction_space_widget
    # convention (see _render_diffraction_image in update_views.py).
    scaled = scale_fn(image).T

    if relevel or first:
        levels = tuple(np.percentile(scaled, [0.1, 99.9]))
        view.setImage(scaled, autoLevels=False, levels=levels, autoRange=first)
    else:
        # Leave levels untouched, so any range the user set by hand on the
        # histogram survives the cursor moving.
        view.setImage(scaled, autoLevels=False, autoRange=False)


class BVMPane(QGroupBox):
    """The Bragg vector map of Bragg vectors saved with the dataset."""

    def __init__(self, bvm, source):
        super().__init__("Bragg Vector Map")

        self.bvm = bvm

        self.view = pg.ImageView()
        self.view.ui.roiBtn.hide()
        self.view.ui.menuBtn.hide()

        # A long path can't wrap, so let it clip rather than widen the
        # column, with the whole of it in the tooltip.
        label = QLabel(source)
        label.setToolTip(source)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(label)
        self.setLayout(layout)

    def set_colormap(self, diffraction_cmap):
        self.view.setColorMap(diffraction_cmap)

    def show_bvm(self, scale_fn, first=False):
        show_diffraction_image(self.view, self.bvm, scale_fn, True, first)


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
    # The pane's remove button was clicked.
    sig_remove_requested = QtCore.pyqtSignal()
    # The g-vector array was switched on or off.
    sig_array_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title):
        super().__init__(title)

        self.apertures = []  # pg.CircleROIs, in the order they were added
        self.rs_selector = None
        self.rs_shape = None
        self.last_dp = None
        self._has_shown_dp = False
        self.vimg = None  # the virtual image from the apertures, once there are any
        self.color = None  # RGB (0-1) chosen for the merged view, if any
        self.g_array = None  # a GVectorArray, once first switched on

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

        self.array_button = QPushButton("g-Vector Array")
        self.array_button.setCheckable(True)
        self.array_button.setToolTip(
            "Apertures on the lattice spanned by two g-vectors -- drag O, g1 "
            "and g2 to fit it to the pattern"
        )
        self.array_button.toggled.connect(self._on_array_toggled)

        # Shown while the array is on: for each g-vector, how many steps the
        # array extends either way along it, and the shift along it (in
        # multiples of it).
        self.n1_spin = QSpinBox()
        self.n2_spin = QSpinBox()
        for spin in (self.n1_spin, self.n2_spin):
            spin.setRange(0, 50)
            spin.setValue(3)
            spin.setPrefix("\u00b1")
            spin.valueChanged.connect(self._on_array_params_changed)
        self.s1_spin = QDoubleSpinBox()
        self.s2_spin = QDoubleSpinBox()
        for spin in (self.s1_spin, self.s2_spin):
            spin.setRange(-1, 1)
            spin.setSingleStep(0.25)
            spin.valueChanged.connect(self._on_array_params_changed)
        self.include_origin_checkbox = QCheckBox("Include origin")
        self.include_origin_checkbox.toggled.connect(self._on_array_params_changed)
        # Only offered when there are Bragg vectors (see set_snap_available).
        self.snap_checkbox = QCheckBox("Snap to Bragg peaks")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setToolTip(
            "Snap O, g1 and g2 onto the nearest Bragg peaks detected in this "
            "pattern -- all three when ticked, and each when dropped near one"
        )
        self.snap_checkbox.toggled.connect(self._on_snap_toggled)
        self.snap_checkbox.hide()

        def vector_label(text, pen):
            label = QLabel(text)
            label.setStyleSheet("color: rgb({}, {}, {});".format(*pen["color"]))
            return label

        array_layout = QGridLayout()
        array_layout.setContentsMargins(0, 0, 0, 0)
        array_layout.setColumnStretch(0, 1)
        array_layout.addWidget(QLabel("repeats"), 0, 2)
        array_layout.addWidget(QLabel("shift"), 0, 3)
        array_layout.addWidget(vector_label("g1", G1_PEN), 1, 1)
        array_layout.addWidget(self.n1_spin, 1, 2)
        array_layout.addWidget(self.s1_spin, 1, 3)
        array_layout.addWidget(self.include_origin_checkbox, 1, 4)
        array_layout.addWidget(vector_label("g2", G2_PEN), 2, 1)
        array_layout.addWidget(self.n2_spin, 2, 2)
        array_layout.addWidget(self.s2_spin, 2, 3)
        array_layout.addWidget(self.snap_checkbox, 2, 4)
        self.array_controls = QWidget()
        self.array_controls.setLayout(array_layout)
        self.array_controls.hide()

        self.remove_button = QToolButton()
        self.remove_button.setText("\u2715")
        self.remove_button.setAutoRaise(True)
        self.remove_button.setToolTip("Remove this preview")
        self.remove_button.clicked.connect(self.sig_remove_requested.emit)

        header = QHBoxLayout()
        header.addStretch()
        header.addWidget(self.array_button)
        header.addWidget(self.clear_apertures_button)
        header.addWidget(self.remove_button)

        self.views_layout = QGridLayout()
        self.views_layout.addWidget(self.dp_view, 0, 0)
        self.views_layout.addWidget(self.vi_view, 0, 1)
        self.views_layout.addWidget(self.dp_label, 1, 0)
        self.views_layout.addWidget(self.vi_label, 1, 1)
        # Pattern and image share the width equally, whatever is under them.
        self.views_layout.setColumnStretch(0, 1)
        self.views_layout.setColumnStretch(1, 1)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self.array_controls)
        layout.addLayout(self.views_layout)
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
        show_diffraction_image(
            self.dp_view, dp, scale_fn, relevel, first=not self._has_shown_dp
        )
        self._has_shown_dp = True

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

        # Nor is one on an array marker.
        if self.array_enabled() and self.g_array.marker_at(ev.scenePos()):
            return

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
        # The click-added apertures, and the array (switched off, keeping its
        # vectors for next time).
        if not self.apertures and not self.array_enabled():
            return
        for roi in self.apertures:
            self.dp_view.getView().removeItem(roi)
        self.apertures = []
        self.array_button.blockSignals(True)
        self.array_button.setChecked(False)
        self.array_button.blockSignals(False)
        self._show_array(False)
        self.sig_apertures_changed.emit()

    # --- g-vector array ---

    def attach_array(self, g_array):
        # Called by the window the first time the array is switched on.
        self.g_array = g_array
        g_array.set_params(**self._array_params(), snap=self.snap_checkbox.isChecked())
        g_array.sig_change_finished.connect(self._on_array_changed)
        self._show_array(self.array_button.isChecked())

    def _on_array_changed(self):
        # Changes made while it's off don't change the virtual image.
        if self.array_enabled():
            self.sig_apertures_changed.emit()

    def array_enabled(self):
        return self.g_array is not None and self.array_button.isChecked()

    def _show_array(self, shown):
        self.array_controls.setVisible(shown)
        if self.g_array is not None:
            self.g_array.set_visible(shown)

    def _on_array_toggled(self, checked):
        self._show_array(checked)
        self.sig_array_toggled.emit(checked)

    def _array_params(self):
        return dict(
            n1=self.n1_spin.value(),
            n2=self.n2_spin.value(),
            s1=self.s1_spin.value(),
            s2=self.s2_spin.value(),
            include_origin=self.include_origin_checkbox.isChecked(),
        )

    def _on_array_params_changed(self, *_):
        if self.g_array is not None:
            self.g_array.set_params(**self._array_params())

    def _on_snap_toggled(self, checked):
        if self.g_array is not None:
            self.g_array.set_params(snap=checked)
            if checked:
                self.g_array.snap_all()

    def set_snap_available(self, available):
        # Snapping needs Bragg vectors saved with the dataset.
        self.snap_checkbox.setVisible(available)
        if self.g_array is not None:
            self.g_array.set_params(snap=available and self.snap_checkbox.isChecked())

    @staticmethod
    def _aperture_geometry(roi):
        radius = roi.size()[0] / 2
        qy = roi.pos()[0] + radius - 0.5
        qx = roi.pos()[1] + radius - 0.5
        return (qx, qy), radius

    def aperture_geometries(self):
        """
        Each aperture as ((qx, qy), radius), in data coordinates: the
        click-added ones, and the array's while it's on.
        """
        geometries = [self._aperture_geometry(roi) for roi in self.apertures]
        if self.array_enabled():
            geometries += self.g_array.apertures()
        return geometries

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


class MergedPane(PreviewPane):
    """
    The previews merged into one: each preview's apertures drawn on a single
    diffraction pattern in that preview's colour, and their virtual images
    combined into a colour composite in the same colours. For viewing only --
    apertures are edited in the separate previews -- apart from the colours,
    which the legend's swatches let the user choose.
    """

    # A legend swatch was clicked: the index of its entry.
    sig_swatch_clicked = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__("Merged Views")

        self.clear_apertures_button.hide()
        self.remove_button.hide()
        self.array_button.hide()
        # The composite has fixed levels, so there's nothing to adjust.
        self.vi_view.ui.histogram.hide()

        # In place of the virtual image label.
        self.vi_label.hide()
        self.legend_layout = QHBoxLayout()
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        legend = QWidget()
        legend.setLayout(self.legend_layout)
        self.views_layout.addWidget(legend, 1, 1)

        self.markers = pg.ScatterPlotItem(pxMode=False)  # sizes in data pixels
        self.dp_view.getView().addItem(self.markers)

    def set_colormaps(self, realspace_cmap, diffraction_cmap):
        # The composite has colours of its own.
        self.dp_view.setColorMap(diffraction_cmap)

    def show_composite(self, rgb):
        # Transposed like show_vi, keeping the colour axis last.
        shown = np.transpose(rgb, (1, 0, 2))
        self.vi_view.setImage(
            shown,
            levels=(0, 1),
            autoRange=self.vi_view.image.shape[:2] != shown.shape[:2],
        )

    def set_aperture_markers(self, apertures):
        """Draw apertures, each ((qx, qy), radius, rgb), as rings in their colour."""
        # Centred where PreviewPane.add_aperture would draw them.
        self.markers.setData(
            [
                {
                    "pos": (qy + 0.5, qx + 0.5),
                    "size": 2 * radius,
                    "pen": pg.mkPen([int(round(255 * c)) for c in rgb], width=2),
                    "brush": None,
                    "symbol": "o",
                }
                for (qx, qy), radius, rgb in apertures
            ]
        )

    def set_legend(self, entries):
        """Show a swatch and name for each entry, (name, rgb)."""
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget() is not None:
                # Deleted later, not now, as this can run from a swatch's own
                # click -- but hidden at once.
                item.widget().hide()
                item.widget().deleteLater()

        self.legend_layout.addWidget(QLabel("Composite:"))
        self.swatches = []
        for i, (name, rgb) in enumerate(entries):
            swatch = QPushButton()
            swatch.setAutoDefault(False)  # see disable_auto_default
            swatch.setFixedSize(18, 18)
            swatch.setCursor(QtCore.Qt.PointingHandCursor)
            swatch.setToolTip(f"Change the colour of {name}")
            hex_color = "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb)
            swatch.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid gray;"
            )
            swatch.clicked.connect(partial(self.sig_swatch_clicked.emit, i))
            self.swatches.append(swatch)
            self.legend_layout.addWidget(swatch)
            self.legend_layout.addWidget(QLabel(name))
        self.legend_layout.addStretch()
