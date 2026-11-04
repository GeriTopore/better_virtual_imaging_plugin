import os
import re
from functools import partial

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.utils import DetectorMode, DetectorShape

from . import bragg, figure, imaging, processing
from .correction_dialog import PixelCorrectionDialog
from .lattice import GVectorArray
from .preview_pane import PANE_MIN_HEIGHT, MergedPane, PreviewPane
from .mask_dialog import MaskDialog
from .save_dialog import SaveFigureDialog
from .wheel import ZOOM_KEY, WheelRouter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer

# The most previews the merged view's list shows before scrolling.
MERGE_LIST_ROWS = 8


class BetterVirtualImagingPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.external.better_virtual_imaging"

    uses_single_action = True
    display_name = "Better Virtual Imaging..."

    def __init__(self, parent: "DataViewer", plugin_action, **kwargs):
        super().__init__()

        self.parent = parent
        self.window = None

        plugin_action.triggered.connect(self.launch_window)
        parent.signal_datacube_changed.connect(self.on_datacube_changed)

    def close(self):
        # Called by py4D-browser on shutdown (since 1.5.3).
        if self.window is not None:
            self.window.close()

    def launch_window(self):
        if self.window is None:
            self.window = BetterVirtualImagingWindow(parent=self.parent)
        else:
            self.window.load_braggvectors()

        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def on_datacube_changed(self):
        # Anything the window computed belongs to the old dataset, so start
        # over with a fresh window -- reopened in the same place if it was
        # open.
        if self.window is None or self.window.keep_through_data_change:
            return
        old_window, self.window = self.window, None
        was_visible = old_window.isVisible()
        geometry = old_window.saveGeometry()
        old_window.close()
        old_window.deleteLater()

        if was_visible:
            self.launch_window()
            self.window.restoreGeometry(geometry)


class BetterVirtualImagingWindow(QDialog):
    def __init__(self, parent: "DataViewer"):
        super().__init__(parent=parent)

        self.parent = parent
        self.panes = []

        # The mean and maximum patterns, computed when first needed.
        self.patterns = {}
        # Set while this window announces a change it made to the dataset's
        # values (not its shape), which it updates itself for, rather than
        # being rebuilt (see BetterVirtualImagingPlugin.on_datacube_changed).
        self.keep_through_data_change = False
        # The folder images were last saved to, to offer again.
        self.save_dir = None

        # The window is rebuilt for every new dataset, but the main window
        # redraws (and emits signal_diffraction_data_changed) for a new one
        # just before announcing it -- so remember which dataset this window
        # belongs to, by shape, rather than trusting parent.datacube.
        datacube = parent.datacube
        self.data_shape = None if datacube is None else datacube.data.shape

        # The real-space cursor follows the main window's (Detector Shape >
        # Virtual Image, and Detector Response > Virtual Image).
        detector = self._main_rs_detector()
        if detector is None:
            self.rs_shape, self.rs_mode = DetectorShape.POINT, DetectorMode.INTEGRATING
        else:
            self.rs_shape, self.rs_mode = detector["shape"], detector["mode"]

        self.setWindowTitle("Better Virtual Imaging")

        self.merge_box = self._build_merge_box()
        apertures_box = self._build_apertures_box()
        scaling_box = self._build_scaling_box()
        binning_box = self._build_binning_box()
        bad_pixels_box = self._build_bad_pixels_box()

        self.add_pane_button = QPushButton("Add Preview")
        self.add_pane_button.clicked.connect(self.add_pane)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.merge_box)
        controls_layout.addWidget(apertures_box)
        controls_layout.addWidget(scaling_box)
        controls_layout.addWidget(binning_box)
        controls_layout.addWidget(bad_pixels_box)
        controls_layout.addStretch()
        self.controls_widget = QWidget()
        self.controls_widget.setLayout(controls_layout)

        # Scrolling vertically if the window's too short for the controls,
        # and never squeezed narrower than they need.
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(self.controls_widget)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(
            self.controls_widget.minimumSizeHint().width()
            + self.controls_scroll.verticalScrollBar().sizeHint().width()
        )

        self.braggvectors = None  # kept for snapping g-vector arrays to peaks

        # Toggles between the separate previews and a merged view of them
        # (built on first use, then hidden rather than deleted).
        self.merged_pane = None
        # The previews that can be merged (those with apertures), and those
        # of them in the merged view (those ticked in merge_list), in order.
        self.mergeable = []
        self.merged = []
        # Every preview's height (see fit_pane_heights).
        self.pane_height = PANE_MIN_HEIGHT
        self.merge_button = QPushButton("Merge Views")
        self.merge_button.setCheckable(True)
        self.merge_button.setToolTip(
            "Combine the previews with apertures into one view, each in its "
            "own colour"
        )
        self.merge_button.toggled.connect(self.on_merge_toggled)

        # The previews, one per row, each a fixed height (see
        # fit_pane_heights), scrolling when there are more than fit -- and
        # kept to the top by the stretch, always last, when there are fewer.
        self.previews_layout = QVBoxLayout()
        self.previews_layout.setContentsMargins(0, 0, 0, 0)
        self.previews_layout.addStretch()
        self.views = QWidget()
        self.views.setLayout(self.previews_layout)
        self.previews_scroll = QScrollArea()
        self.previews_scroll.setWidgetResizable(True)
        self.previews_scroll.setWidget(self.views)
        self.previews_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Always on, so the previews' width (and so their height) doesn't
        # change with whether they scroll.
        self.previews_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.previews_scroll.viewport().installEventFilter(self)
        # The wheel scrolls the previews even over their images, zooming
        # those only with Cmd/Ctrl held (see wheel.py).
        self.wheel_router = WheelRouter(self.previews_scroll, parent=self)
        wheel_hint = QLabel(
            f"Scroll to move through the previews, {ZOOM_KEY} + scroll to zoom"
        )
        wheel_hint.setEnabled(False)  # greyed, as a hint

        previews_header = QHBoxLayout()
        previews_header.addWidget(self.add_pane_button)
        previews_header.addWidget(self.merge_button)
        previews_header.addStretch()
        previews_header.addWidget(wheel_hint)
        previews_layout = QVBoxLayout()
        previews_layout.setContentsMargins(0, 0, 0, 0)
        previews_layout.addLayout(previews_header)
        previews_layout.addWidget(self.previews_scroll)
        previews_column = QWidget()
        previews_column.setLayout(previews_layout)

        self.main_splitter = QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(self.controls_scroll)
        self.main_splitter.addWidget(previews_column)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setSizes([300, 1100])

        layout = QVBoxLayout(self)
        layout.addWidget(self.main_splitter)

        self.resize(1400, 800)

        # Previews use the same colormaps as the main window's views, and
        # follow them if they're changed there (via the gradient's
        # right-click menu).
        for widget in (parent.real_space_widget, parent.diffraction_space_widget):
            widget.ui.histogram.gradient.sigGradientChangeFinished.connect(
                self.sync_colormaps
            )
        # Fires whenever the main window redraws its diffraction pattern,
        # which includes changes to its real-space detector shape/response.
        parent.signal_diffraction_data_changed.connect(self.on_main_diffraction_changed)

        self._append_pane()
        self.load_braggvectors()

        self.disable_auto_default(self)

    def disable_auto_default(self, widget):
        # QDialog makes every push button in it "auto-default", drawing the
        # first one highlighted as if already chosen, and having Enter in any
        # spinbox click it. Needs calling again for buttons added later
        # (e.g. new preview panes).
        for button in widget.findChildren(QPushButton):
            button.setAutoDefault(False)

    def _build_apertures_box(self):
        box = QGroupBox("Virtual Apertures")

        self.add_on_click_button = QPushButton("Add Aperture on Click")
        self.add_on_click_button.setCheckable(True)
        self.add_on_click_button.toggled.connect(self.on_click_to_add_toggled)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setDecimals(1)
        self.radius_spin.setSingleStep(0.5)
        if self.data_shape is None:
            self.radius_spin.setRange(0.5, 1000)
            self.radius_spin.setValue(5)
        else:
            Q_Nx, Q_Ny = self.data_shape[2:]
            self.radius_spin.setRange(0.5, max(Q_Nx, Q_Ny))
            self.radius_spin.setValue(max(1, round(min(Q_Nx, Q_Ny) / 40)))
        self.radius_spin.valueChanged.connect(self._update_array_radii)

        # Locks new apertures to the bright-field disk radius.
        self.bf_disk = None  # (radius, qx0, qy0), measured when first needed
        self.use_bf_radius_checkbox = QCheckBox("Use BF radius")
        self.use_bf_radius_checkbox.setToolTip(
            "Size new apertures to the bright-field disk, as measured from the "
            "mean diffraction pattern"
        )
        self.use_bf_radius_checkbox.toggled.connect(self.on_use_bf_radius_toggled)

        radius_form = QFormLayout()
        radius_form.addRow("Radius (px)", self.radius_spin)
        radius_form.addRow(self.use_bf_radius_checkbox)

        layout = QVBoxLayout()
        layout.addWidget(self.add_on_click_button)
        layout.addLayout(radius_form)
        box.setLayout(layout)
        return box

    def _build_scaling_box(self):
        scaling_box = QGroupBox("Diffraction Display")
        scaling_layout = QVBoxLayout()

        self.scaling_group = QButtonGroup(self)
        self.linear_radio = QRadioButton("Linear")
        self.log_radio = QRadioButton("Log")
        self.power_radio = QRadioButton("Power")
        self.log_radio.setChecked(True)
        for button in (self.linear_radio, self.log_radio, self.power_radio):
            self.scaling_group.addButton(button)
            scaling_layout.addWidget(button)
            # only redraw once, when a button becomes checked -- QButtonGroup
            # toggles the old and new selection in the same click
            button.toggled.connect(
                lambda checked: self.on_scaling_changed() if checked else None
            )

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Power"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 2.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(0.5)
        self.gamma_spin.valueChanged.connect(self.on_scaling_changed)
        gamma_row.addWidget(self.gamma_spin)
        scaling_layout.addLayout(gamma_row)

        # Enabled once there are Bragg vectors (see load_braggvectors).
        self.show_peaks_checkbox = QCheckBox("Show detected peaks")
        self.show_peaks_checkbox.setEnabled(False)
        self.show_peaks_checkbox.setToolTip(
            "Mark the Bragg peaks detected at each preview's cursor (every scan "
            "position in a rectangle) -- needs Bragg vectors saved with the "
            "dataset"
        )
        self.show_peaks_checkbox.toggled.connect(lambda: self.update_peaks())
        scaling_layout.addWidget(self.show_peaks_checkbox)

        scaling_box.setLayout(scaling_layout)
        return scaling_box

    def _build_binning_box(self):
        box = QGroupBox("Binning")

        self.shape_label = QLabel()
        self.shape_label.setWordWrap(True)
        self._update_shape_label()

        # Bin factor and button, for diffraction space (Q) and real space (R).
        self.bin_spins = {}
        layout = QGridLayout()
        layout.addWidget(self.shape_label, 0, 0, 1, 3)
        rows = (
            ("Q", "Diffraction", "pixels in every pattern"),
            ("R", "Scan", "scan positions"),
        )
        for row, (axis, name, blocks) in enumerate(rows, 1):
            spin = QSpinBox()
            spin.setRange(2, 2 if self.data_shape is None else self._max_bin(axis))
            spin.setPrefix("\u00d7")
            button = QPushButton(f"Bin {axis}")
            button.setToolTip(
                f"Sum each N x N block of {blocks} (py4DSTEM's bin_{axis}), in "
                "the main window's dataset too"
            )
            button.clicked.connect(partial(self.on_bin, axis))
            button.setEnabled(self.data_shape is not None)
            self.bin_spins[axis] = spin
            layout.addWidget(QLabel(name), row, 0)
            layout.addWidget(spin, row, 1)
            layout.addWidget(button, row, 2)
        box.setLayout(layout)
        return box

    def _max_bin(self, axis):
        shape = self.data_shape[2:] if axis == "Q" else self.data_shape[:2]
        return max(2, min(shape))

    def _update_shape_label(self):
        if self.data_shape is None:
            self.shape_label.setText("No data")
            return
        R_Nx, R_Ny, Q_Nx, Q_Ny = self.data_shape
        self.shape_label.setText(
            f"Scan {R_Nx} \u00d7 {R_Ny}, patterns {Q_Nx} \u00d7 {Q_Ny}"
        )

    def _build_merge_box(self):
        # Shown only while the views are merged (see _fill_merge_list).
        box = QGroupBox("Merged Previews")
        self.merge_list = QListWidget()
        self.merge_list.setToolTip(
            "Untick a preview to leave it out of the merged view (previews "
            "without apertures can't be merged)"
        )
        self.merge_list.itemChanged.connect(self.on_merge_list_changed)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Show in the merged view:"))
        layout.addWidget(self.merge_list)
        box.setLayout(layout)
        box.hide()
        return box

    def _build_bad_pixels_box(self):
        box = QGroupBox("Bad Pixels")

        self.hot_thresh_spin = QDoubleSpinBox()
        self.hot_thresh_spin.setDecimals(2)
        self.hot_thresh_spin.setRange(0.01, 1000)
        self.hot_thresh_spin.setSingleStep(0.1)
        self.hot_thresh_spin.setValue(1.0)
        self.hot_thresh_spin.setSuffix(" \u00d7 mean")
        self.hot_thresh_spin.setToolTip(
            "A pixel is hot if, in the mean pattern, it's brighter than the "
            "second brightest pixel around it by more than this many times the "
            "pattern's average intensity (py4DSTEM's filter_hot_pixels)"
        )
        hot_button = QPushButton("Filter Hot Pixels")
        hot_button.clicked.connect(partial(self.on_fix_pixels, "hot"))

        self.dead_fraction_spin = QDoubleSpinBox()
        self.dead_fraction_spin.setDecimals(2)
        self.dead_fraction_spin.setRange(0.01, 1.0)
        self.dead_fraction_spin.setSingleStep(0.05)
        self.dead_fraction_spin.setValue(0.1)
        self.dead_fraction_spin.setSuffix(" \u00d7 neighbours")
        self.dead_fraction_spin.setToolTip(
            "A pixel is dead if, in the mean pattern, it's dimmer than this "
            "fraction of the median of the 8 pixels around it"
        )
        dead_button = QPushButton("Fix Dead Pixels")
        dead_button.clicked.connect(partial(self.on_fix_pixels, "dead"))

        for button in (hot_button, dead_button):
            button.setEnabled(self.data_shape is not None)

        self.pixels_status = QLabel()
        self.pixels_status.setWordWrap(True)

        note = QLabel(
            "Each is replaced, in every pattern, by the median of the pixels "
            "around it."
        )
        note.setWordWrap(True)

        layout = QGridLayout()
        layout.addWidget(QLabel("Hot above"), 0, 0)
        layout.addWidget(self.hot_thresh_spin, 0, 1)
        layout.addWidget(hot_button, 1, 0, 1, 2)
        layout.addWidget(QLabel("Dead below"), 2, 0)
        layout.addWidget(self.dead_fraction_spin, 2, 1)
        layout.addWidget(dead_button, 3, 0, 1, 2)
        layout.addWidget(note, 4, 0, 1, 2)
        layout.addWidget(self.pixels_status, 5, 0, 1, 2)
        box.setLayout(layout)
        return box

    def get_scaling_fn(self):
        gamma = self.gamma_spin.value()
        if self.linear_radio.isChecked():
            return lambda dp: dp.astype(np.float64, copy=False)
        elif self.log_radio.isChecked():
            return lambda dp: np.log(np.maximum(dp, 1e-6))
        else:
            return lambda dp: np.power(np.maximum(dp, 0), gamma)

    # --- Panes ---

    def _append_pane(self):
        pane = PreviewPane(f"Preview {len(self.panes) + 1}")
        pane.sig_scan_selection_changed.connect(partial(self.update_dps, panes=[pane]))
        pane.sig_apertures_changed.connect(partial(self.update_vis, panes=[pane]))
        pane.sig_dp_clicked.connect(partial(self.on_dp_clicked, pane))
        pane.sig_remove_requested.connect(partial(self.remove_pane, pane))
        pane.sig_array_toggled.connect(partial(self.on_array_toggled, pane))
        pane.sig_pattern_mode_changed.connect(
            partial(self.update_dps, panes=[pane], relevel=True)
        )
        pane.sig_save_requested.connect(partial(self.on_save_image, pane))
        pane.sig_save_mask_requested.connect(partial(self.on_save_mask, pane))
        pane.set_snap_available(self.braggvectors is not None)
        pane.set_click_to_add(self.add_on_click_button.isChecked())
        self.panes.append(pane)
        self._layout_panes()
        self.disable_auto_default(pane)
        self.wheel_router.add_pane(pane)
        self.sync_colormaps(panes=[pane])

        self._place_cursor_from_main(pane)
        self.update_vis(panes=[pane])
        self.update_dps(panes=[pane])

        self._update_pane_buttons()
        # Once it's laid out, size it, and scroll to it if it's below the
        # others.
        QtCore.QTimer.singleShot(0, lambda: self._fit_and_show(pane))

    def _fit_and_show(self, pane):
        self.fit_pane_heights()
        self.previews_scroll.ensureWidgetVisible(pane)

    def add_pane(self):
        self._append_pane()

    def remove_pane(self, pane):
        if len(self.panes) <= 1:
            return
        self.panes.remove(pane)
        self.wheel_router.remove_pane(pane)
        self.previews_layout.removeWidget(pane)
        pane.hide()
        pane.deleteLater()

        # Keep the previews numbered in order.
        for i, remaining in enumerate(self.panes):
            remaining.set_default_title(f"Preview {i + 1}")
        self._layout_panes()
        self._update_pane_buttons()

    def _update_pane_buttons(self):
        # No adding while merged; no removing the last one.
        self.add_pane_button.setEnabled(not self.merge_button.isChecked())
        for pane in self.panes:
            pane.remove_button.setVisible(len(self.panes) > 1)

    def _layout_panes(self):
        # One per row, in order.
        for pane in self.panes:
            self.previews_layout.removeWidget(pane)
        for pane in self.panes:
            self._add_to_previews(pane)

    def _add_to_previews(self, pane):
        # Below the others, above the stretch.
        pane.setFixedHeight(self.pane_height)
        self.previews_layout.insertWidget(self.previews_layout.count() - 1, pane)

    def fit_pane_heights(self):
        """
        Make every preview as tall as the space they're shown in, as one
        alone would be -- but no taller than makes its images square, beyond
        which they'd only gain space around them, nor shorter than
        PANE_MIN_HEIGHT. More than fit scroll.
        """
        height = self.previews_scroll.viewport().height()
        shown = [pane for pane in self._cursor_panes() if pane.isVisible()]
        image_heights = [pane.height_for_images() for pane in shown]
        image_heights = [h for h in image_heights if h is not None]
        if image_heights:
            height = min(height, max(image_heights))
        self.pane_height = max(height, PANE_MIN_HEIGHT)
        for pane in self._cursor_panes():
            pane.setFixedHeight(self.pane_height)

    def eventFilter(self, obj, event):
        if (
            obj is self.previews_scroll.viewport()
            and event.type() == QtCore.QEvent.Resize
        ):
            self.fit_pane_heights()
        return super().eventFilter(obj, event)

    def _cursor_panes(self):
        # Every pane with a diffraction pattern and real-space cursor: the
        # previews, and the merged view once there is one.
        if self.merged_pane is None:
            return self.panes
        return self.panes + [self.merged_pane]

    def sync_colormaps(self, *_, panes=None):
        parent = self.parent
        realspace_cmap = parent.real_space_widget.ui.histogram.gradient.colorMap()
        diffraction_cmap = (
            parent.diffraction_space_widget.ui.histogram.gradient.colorMap()
        )
        for pane in panes if panes is not None else self._cursor_panes():
            pane.set_colormaps(realspace_cmap, diffraction_cmap)

    # --- Real-space cursor ---

    def _has_data(self):
        datacube = self.parent.datacube
        return datacube is not None and datacube.data.shape == self.data_shape

    def _main_rs_detector(self):
        if not self._has_data():
            return None
        return self.parent.get_virtual_image_detector()

    def _rect_around(self, xc, yc, like=None):
        """
        Slices for a rectangle centred on (xc, yc), sized like the `like`
        slices (the main window's selection) or, failing that, the main
        window's default, and kept on the scan.
        """
        R_Nx, R_Ny = self.data_shape[:2]
        nx = ny = 0
        if like is not None:
            nx, ny = like[0].stop - like[0].start, like[1].stop - like[1].start
        if nx <= 0 or ny <= 0:
            nx = ny = max(1, int(min(R_Nx, R_Ny) / 10))
        nx, ny = min(nx, R_Nx), min(ny, R_Ny)
        x0 = int(np.clip(xc - nx // 2, 0, R_Nx - nx))
        y0 = int(np.clip(yc - ny // 2, 0, R_Ny - ny))
        return slice(x0, x0 + nx), slice(y0, y0 + ny)

    def _place_cursor_from_main(self, pane):
        # Start a new pane's cursor where the main window's is.
        detector = self._main_rs_detector()
        if detector is None:
            pane.set_point_selector(0, 0)
        elif detector["shape"] is DetectorShape.POINT:
            pane.set_point_selector(*detector["point"])
        else:
            slice_x, slice_y = detector["slice"]
            if slice_x.start >= slice_x.stop or slice_y.start >= slice_y.stop:
                # The main window's rectangle is off the scan.
                R_Nx, R_Ny = self.data_shape[:2]
                slice_x, slice_y = self._rect_around(R_Nx // 2, R_Ny // 2)
            pane.set_rect_selector(slice_x, slice_y)

    def on_main_diffraction_changed(self):
        detector = self._main_rs_detector()
        if detector is None:
            return
        shape, mode = detector["shape"], detector["mode"]
        if shape is self.rs_shape and mode is self.rs_mode:
            return

        if shape is not self.rs_shape:
            # Swap each pane's cursor for the new shape, in the same place.
            Rshape = self.data_shape[:2]
            for pane in self._cursor_panes():
                xc, yc = pane.scan_centre(Rshape)
                if shape is DetectorShape.POINT:
                    pane.set_point_selector(xc, yc)
                else:
                    pane.set_rect_selector(
                        *self._rect_around(xc, yc, like=detector["slice"])
                    )

        self.rs_shape, self.rs_mode = shape, mode
        self.update_dps()

    # --- Computing and showing ---

    def update_dps(self, panes=None, relevel=False):
        # The diffraction pattern at each pane's cursor.
        if not self._has_data():
            return

        data = self.parent.datacube.data
        Rshape = self.data_shape[:2]
        scale_fn = self.get_scaling_fn()

        for pane in panes if panes is not None else self._cursor_panes():
            mode = pane.pattern_mode()
            if mode != "cursor":
                name = "Mean" if mode == "mean" else "Maximum"
                pane.dp_label.setText(f"Diffraction: {name} of all patterns")
                pane.show_dp(self.overall_pattern(mode), scale_fn, relevel=relevel)
                continue

            selection = pane.get_scan_selection(Rshape)
            if pane.rs_shape is DetectorShape.POINT:
                xc, yc = selection
                dp = np.asarray(data[xc, yc])
                pane.dp_label.setText(f"Diffraction: Point [{xc},{yc}]")
            else:
                slice_x, slice_y = selection
                if slice_x.start >= slice_x.stop or slice_y.start >= slice_y.stop:
                    pane.dp_label.setText("Diffraction: cursor is off the scan")
                    continue
                if self.rs_mode is DetectorMode.MAXIMUM:
                    dp = np.max(data[slice_x, slice_y], axis=(0, 1))
                    label = "Maximum"
                else:
                    dp = np.mean(data[slice_x, slice_y], axis=(0, 1))
                    label = "Mean"
                pane.dp_label.setText(
                    f"Diffraction: {label} of [{slice_x.start}:{slice_x.stop},"
                    f"{slice_y.start}:{slice_y.stop}]"
                )
            pane.show_dp(dp, scale_fn, relevel=relevel)
        self.update_peaks(panes)

    def overall_pattern(self, kind):
        """The "mean" or "max" of every pattern in the scan, computed once."""
        if kind not in self.patterns:
            data = self.parent.datacube.data
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                if kind == "mean":
                    self.patterns[kind] = imaging.mean_dp(data)
                else:
                    self.patterns[kind] = imaging.max_dp(data)
            finally:
                QApplication.restoreOverrideCursor()
        return self.patterns[kind]

    def update_peaks(self, panes=None):
        # The detected peaks in each pane's pattern, if they're to be shown.
        show = self.show_peaks_checkbox.isChecked()
        for pane in panes if panes is not None else self._cursor_panes():
            pane.set_peaks(self._bragg_peaks(pane) if show else None)

    def update_vis(self, panes=None):
        # The virtual image from each pane's apertures -- or, until it has
        # any, the main window's virtual image, to navigate by.
        if not self._has_data():
            return

        datacube = self.parent.datacube
        Rshape = self.data_shape[:2]

        for pane in panes if panes is not None else self.panes:
            apertures = pane.aperture_geometries()
            if not apertures:
                vimg = self.parent.unscaled_realspace_image
                if vimg is None or vimg.shape != Rshape:
                    vimg = np.zeros(Rshape)
                elif np.iscomplexobj(vimg):  # the main window's CoM mode
                    vimg = np.abs(vimg)
                pane.vimg = None
                pane.show_vi(vimg)
                pane.vi_label.setText("Virtual Image: main window's (no apertures)")
                continue

            mask = imaging.aperture_mask(datacube.Qshape, apertures)
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                vimg = imaging.masked_sum(datacube.data, mask)
            finally:
                QApplication.restoreOverrideCursor()
            pane.vimg = vimg
            pane.show_vi(vimg)
            n = len(apertures)
            pane.vi_label.setText(f"Virtual Image: {n} aperture{'' if n == 1 else 's'}")

    # --- User actions ---

    def on_click_to_add_toggled(self, checked):
        for pane in self.panes:
            pane.set_click_to_add(checked)

    def on_dp_clicked(self, pane, qx, qy):
        if not self.add_on_click_button.isChecked() or not self._has_data():
            return
        pane.add_aperture(qx, qy, self.aperture_radius())
        self.update_vis(panes=[pane])

    def aperture_radius(self):
        # The spinbox only shows the measured radius rounded, so use it as is.
        if self.use_bf_radius_checkbox.isChecked():
            return self.bf_disk[0]
        return self.radius_spin.value()

    def on_use_bf_radius_toggled(self, checked):
        if checked and self._measure_bf_disk() is None:
            # Leave the radius under the user's control.
            self.use_bf_radius_checkbox.setChecked(False)
            return
        if checked:
            self.radius_spin.setValue(self.bf_disk[0])
        self.radius_spin.setEnabled(not checked)
        self._update_array_radii()

    def _measure_bf_disk(self):
        # Measured once, from the mean pattern; None if that fails.
        if self.bf_disk is not None or not self._has_data():
            return self.bf_disk
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.bf_disk = imaging.measure_bf_disk(self.overall_pattern("mean"))
        except Exception as exc:
            self.parent.statusBar().showMessage(
                f"Couldn't measure the BF disk: {exc}", 5_000
            )
        finally:
            QApplication.restoreOverrideCursor()
        return self.bf_disk

    # --- g-vector arrays ---

    def on_array_toggled(self, pane, checked):
        if checked and pane.g_array is None and self._has_data():
            # Start at the BF disk's centre (the pattern's, failing that),
            # with g-vectors along qx and qy for the user to drag into place.
            Q_Nx, Q_Ny = self.data_shape[2:]
            bf_disk = self._measure_bf_disk()
            origin = bf_disk[1:] if bf_disk is not None else (Q_Nx / 2, Q_Ny / 2)
            length = round(min(Q_Nx, Q_Ny) / 6)
            pane.attach_array(
                GVectorArray(
                    pane.dp_view.getView(),
                    (Q_Nx, Q_Ny),
                    origin,
                    g1=(length, 0),
                    g2=(0, length),
                    radius=self.aperture_radius(),
                    peaks_fn=partial(self._bragg_peaks, pane),
                )
            )
            pane.set_snap_available(self.braggvectors is not None)
            # Start on the pattern's peaks, if snapping's on (and possible).
            if pane.g_array.snap:
                pane.g_array.snap_all(emit=False)
        self.update_vis(panes=[pane])

    def _bragg_peaks(self, pane):
        # The Bragg peaks detected in the pattern a pane shows: at its
        # cursor's scan position, or at every one in its rectangle.
        if self.braggvectors is None or not self._has_data():
            return None
        selection = pane.get_scan_selection(self.data_shape[:2])
        if pane.rs_shape is DetectorShape.POINT:
            positions = [selection]
        else:
            slice_x, slice_y = selection
            positions = [
                (rx, ry)
                for rx in range(slice_x.start, slice_x.stop)
                for ry in range(slice_y.start, slice_y.stop)
            ]
        return bragg.peaks_at(self.braggvectors, positions)

    def _update_array_radii(self, *_):
        # Arrays follow the radius setting (unlike click-added apertures,
        # which keep the size they were added at).
        radius = self.aperture_radius()
        for pane in self.panes:
            if pane.g_array is not None and pane.g_array.radius != radius:
                pane.g_array.set_params(radius=radius)

    def on_scaling_changed(self, *_):
        # Redisplay each pane's cached pattern with fresh levels for the new
        # scaling.
        scale_fn = self.get_scaling_fn()
        for pane in self._cursor_panes():
            if pane.last_dp is not None:
                pane.show_dp(pane.last_dp, scale_fn, relevel=True)

    # --- Merged view ---

    def on_merge_toggled(self, checked):
        if not checked:
            self._separate_views()
        elif not self._merge_views():
            self.merge_button.blockSignals(True)
            self.merge_button.setChecked(False)
            self.merge_button.blockSignals(False)

    def _merge_views(self):
        # Only previews with apertures have a virtual image of their own to
        # merge; the rest show the main window's.
        mergeable = [pane for pane in self.panes if pane.vimg is not None]
        if not mergeable or not self._has_data():
            self.parent.statusBar().showMessage(
                "Add apertures to a preview to merge its view", 5_000
            )
            return False

        if self.merged_pane is None:
            self.merged_pane = MergedPane()
            self.merged_pane.sig_scan_selection_changed.connect(
                partial(self.update_dps, panes=[self.merged_pane])
            )
            self.merged_pane.sig_swatch_clicked.connect(self.on_swatch_clicked)
            self.merged_pane.sig_pattern_mode_changed.connect(
                partial(self.update_dps, panes=[self.merged_pane], relevel=True)
            )
            self.merged_pane.sig_save_requested.connect(
                partial(self.on_save_image, self.merged_pane)
            )
            self.sync_colormaps(panes=[self.merged_pane])
            self.wheel_router.add_pane(self.merged_pane)

        # Start the cursor where the first merged preview's is.
        Rshape = self.data_shape[:2]
        source = mergeable[0]
        xc, yc = source.scan_centre(Rshape)
        if source.rs_shape is DetectorShape.POINT:
            self.merged_pane.set_point_selector(xc, yc)
        else:
            like = source.get_scan_selection(Rshape)
            self.merged_pane.set_rect_selector(*self._rect_around(xc, yc, like=like))

        self.mergeable = mergeable
        self.merged = [pane for pane in mergeable if pane.in_merge]
        self._fill_merge_list()
        self.merge_box.show()
        self._show_merged()

        for pane in self.panes:
            self.previews_layout.removeWidget(pane)
            pane.hide()
        self._add_to_previews(self.merged_pane)
        self.merged_pane.show()
        QtCore.QTimer.singleShot(0, self.fit_pane_heights)
        self.update_dps(panes=[self.merged_pane])

        self._update_pane_buttons()
        return True

    def _fill_merge_list(self):
        # Every preview, ticked if it's in the merged view; those without
        # apertures greyed out.
        self.merge_list.blockSignals(True)
        self.merge_list.clear()
        for i, pane in enumerate(self.panes):
            item = QListWidgetItem(pane.title())
            item.setData(QtCore.Qt.UserRole, i)
            if pane in self.mergeable:
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                checked = pane in self.merged
                item.setCheckState(
                    QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
                )
            else:
                item.setText(f"{pane.title()} (no apertures)")
                item.setFlags(QtCore.Qt.NoItemFlags)
                item.setCheckState(QtCore.Qt.Unchecked)
            self.merge_list.addItem(item)
        self.merge_list.blockSignals(False)
        # Only as tall as its rows, up to MERGE_LIST_ROWS of them.
        rows = min(max(self.merge_list.count(), 1), MERGE_LIST_ROWS)
        self.merge_list.setFixedHeight(
            rows * self.merge_list.sizeHintForRow(0) + 2 * self.merge_list.frameWidth()
        )

    def on_merge_list_changed(self, item):
        pane = self.panes[item.data(QtCore.Qt.UserRole)]
        pane.in_merge = item.checkState() == QtCore.Qt.Checked
        self.merged = [pane for pane in self.mergeable if pane.in_merge]
        self._show_merged()

    def _merged_colors(self):
        # Each merged preview's chosen colour, or else its default -- the
        # defaults spread over every preview that can be merged, so each
        # keeps its colour whichever others are ticked.
        defaults = imaging.composite_colors(len(self.mergeable))
        return [
            (
                pane.color
                if pane.color is not None
                else defaults[self.mergeable.index(pane)]
            )
            for pane in self.merged
        ]

    def _show_merged(self):
        colors = self._merged_colors()
        if self.merged:
            composite = imaging.color_composite(
                [pane.vimg for pane in self.merged], colors
            )
        else:
            composite = np.zeros(tuple(self.data_shape[:2]) + (3,))
        self.merged_pane.show_composite(composite)
        self.merged_pane.set_aperture_markers(
            [
                (centre, radius, color)
                for pane, color in zip(self.merged, colors)
                for centre, radius in pane.aperture_geometries()
            ]
        )
        self.merged_pane.set_legend(
            [(pane.title(), color) for pane, color in zip(self.merged, colors)]
        )

    def on_swatch_clicked(self, index):
        pane = self.merged[index]
        dialog = QColorDialog(QColor.fromRgbF(*self._merged_colors()[index]), self)
        dialog.setWindowTitle(f"Colour for {pane.title()}")
        # Qt's own dialog rather than the native macOS colour panel, which
        # can't be placed, and which on closing hands focus to the main
        # window -- burying this one behind it.
        dialog.setOption(QColorDialog.DontUseNativeDialog)
        self._place_next_to(dialog, self.merged_pane.swatches[index])
        accepted = dialog.exec_() == QDialog.Accepted
        self.raise_()
        self.activateWindow()
        if not accepted:
            return
        pane.color = dialog.selectedColor().getRgbF()[:3]
        self._show_merged()

    def _place_next_to(self, dialog, widget):
        # Just below the widget, or above it if there isn't room, kept on
        # the widget's screen.
        dialog.adjustSize()
        # Until it's shown, the dialog's frame doesn't include its title bar;
        # take that from this window's.
        title_bar = self.frameGeometry().height() - self.geometry().height()
        width = dialog.frameGeometry().width()
        height = dialog.frameGeometry().height() + title_bar
        screen = widget.screen().availableGeometry()
        below = widget.mapToGlobal(QtCore.QPoint(0, widget.height()))
        above_y = widget.mapToGlobal(QtCore.QPoint(0, 0)).y() - height
        y = below.y() if below.y() + height <= screen.bottom() else above_y
        x = min(max(below.x(), screen.left()), screen.right() - width)
        y = min(max(y, screen.top()), screen.bottom() - height)
        dialog.move(x, y)

    def _separate_views(self):
        self.merge_box.hide()
        self.previews_layout.removeWidget(self.merged_pane)
        self.merged_pane.hide()
        for pane in self.panes:
            pane.show()
        self._layout_panes()
        self._update_pane_buttons()
        QtCore.QTimer.singleShot(0, self.fit_pane_heights)

    # --- Saving images ---

    def _dataset_name(self):
        # The loaded file's name (without extension) and folder, if known.
        filepath = bragg.loaded_filepath(self.parent)
        if filepath is None:
            return "dataset", None
        name = os.path.splitext(os.path.basename(filepath))[0]
        return name, os.path.dirname(filepath)

    def on_save_image(self, pane):
        if pane.last_dp is None:
            return
        name, data_dir = self._dataset_name()
        directory = self.save_dir or data_dir or os.path.expanduser("~")
        if pane is self.merged_pane:
            suffix = "merged"
            vi_title = "Virtual Images: colour composite"
        else:
            suffix = self._file_suffix(pane)
            vi_title = pane.vi_label.text()

        fig = figure.pane_figure(
            pane, f"{name}: {pane.title()}", pane.dp_label.text(), vi_title
        )
        dialog = SaveFigureDialog(
            fig, os.path.join(directory, f"{name}_VI_{suffix}.png"), parent=self
        )
        if dialog.exec_() == QDialog.Accepted:
            self.save_dir = os.path.dirname(dialog.saved_path)
            self.parent.statusBar().showMessage(f"Saved {dialog.saved_path}", 5_000)

    def _file_suffix(self, pane):
        # The user's name for the preview, made safe for a file name, or
        # else its number.
        custom = pane.custom_name and re.sub(r"[^\w.-]+", "_", pane.custom_name)
        suffix = custom.strip("_.") if custom else ""
        return suffix or f"preview_{self.panes.index(pane) + 1}"

    def on_save_mask(self, pane):
        # The virtual image as shown -- the preview's own, or the main
        # window's if it has no apertures -- transposed back.
        image = pane.vi_view.image
        if image is None or image.ndim != 2 or not self._has_data():
            return
        name, data_dir = self._dataset_name()
        directory = self.save_dir or data_dir or os.path.expanduser("~")
        dialog = MaskDialog(
            image.T,
            pane.vi_view.ui.histogram.gradient.colorMap(),
            os.path.join(directory, f"{name}_VI_mask_{self._file_suffix(pane)}.npy"),
            threshold=pane.mask_threshold,
            invert=pane.mask_invert,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            pane.mask_threshold, pane.mask_invert = dialog.threshold(), dialog.invert()
            self.save_dir = os.path.dirname(dialog.saved_path)
            self.parent.statusBar().showMessage(f"Saved {dialog.saved_path}", 5_000)

    # --- Bragg vectors ---

    def load_braggvectors(self):
        # Read any Bragg vectors saved with the dataset, for showing detected
        # peaks and snapping g-vector arrays to them. Checked again whenever
        # the window is reopened, as they may have been saved since (e.g. by
        # the Disk Detection plugin).
        if self.braggvectors is not None or not self._has_data():
            return
        source = bragg.find_braggvectors(self.parent)
        if source is None:
            return
        filepath, datapath = source

        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            braggvectors = bragg.read_braggvectors(filepath, datapath)
        except Exception as exc:
            self.parent.statusBar().showMessage(
                f"Couldn't read the Bragg vectors at {datapath}: {exc}", 5_000
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.braggvectors = braggvectors
        for pane in self.panes:
            pane.set_snap_available(True)
        self.show_peaks_checkbox.setEnabled(True)

    # --- Processing the dataset ---

    def on_bin(self, axis):
        if not self._has_data():
            return
        datacube = self.parent.datacube
        if not processing.can_bin(datacube.data):
            QMessageBox.information(
                self,
                "Bin Data",
                "Memory-mapped HDF5 data can't be binned here. Load the file "
                "without memory mapping, or use File > Load Data Binned.",
            )
            return

        N = self.bin_spins[axis].value()
        shape = self.data_shape[2:] if axis == "Q" else self.data_shape[:2]
        new_shape = tuple(n // N for n in shape)
        what = "diffraction patterns" if axis == "Q" else "scan"
        unit = "pixels" if axis == "Q" else "positions"
        cropped = any(n % N for n in shape)
        answer = QMessageBox.question(
            self,
            "Bin Data",
            f"Bin the {what} by {N}, from {shape[0]} \u00d7 {shape[1]} to "
            f"{new_shape[0]} \u00d7 {new_shape[1]} {unit}"
            + (" (dropping the leftover rows/columns)" if cropped else "")
            + "?\n\nThis changes the dataset in the main window too, and starts "
            "this window afresh. Reload the file to undo it.",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if answer != QMessageBox.Ok:
            return

        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            processing.bin_datacube(datacube, axis, N)
        except Exception as exc:
            QMessageBox.warning(self, "Bin Data", f"Binning failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        # Announced as a new dataset, keeping the window title (which other
        # plugins check against the data's shape before trusting it as the
        # data's file). The plugin rebuilds this window for it.
        self.parent.set_datacube(datacube, self.parent.windowTitle())
        self.parent.statusBar().showMessage(
            f"Binned {axis} by {N}: data now {' x '.join(map(str, datacube.data.shape))}",
            5_000,
        )

    def on_fix_pixels(self, kind):
        if not self._has_data():
            return
        datacube = self.parent.datacube
        if not processing.can_modify(datacube.data):
            self.pixels_status.setText(
                "Memory-mapped data can't be changed here -- load the file "
                "without memory mapping."
            )
            return

        before = self.overall_pattern("mean")
        if kind == "hot":
            factor = self.hot_thresh_spin.value()
            thresh = factor * float(np.mean(before))
            mask = processing.hot_pixel_mask(before, thresh)
            criterion = (
                f"more than {thresh:.3g} counts ({factor:g} \u00d7 the average) "
                "brighter than the second brightest pixel around them"
            )
        else:
            fraction = self.dead_fraction_spin.value()
            mask = processing.dead_pixel_mask(before, fraction)
            criterion = (
                f"dimmer than {fraction:g} \u00d7 the median of the pixels around them"
            )
        n = int(mask.sum())
        plural = "" if n == 1 else "s"
        if n == 0:
            self.pixels_status.setText(f"No {kind} pixels found.")
            return

        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            original = processing.replace_pixels(datacube, mask)
            after = before.copy()
            after[mask] = processing.masked_mean(datacube.data, mask)
        finally:
            QApplication.restoreOverrideCursor()

        dialog = PixelCorrectionDialog(
            f"{kind.capitalize()} Pixels",
            f"Found {n} {kind} pixel{plural} in the mean pattern -- {criterion} "
            f"(ringed in red) -- and replaced {'it' if n == 1 else 'them'} in every "
            "pattern with the median of the "
            f"{processing.KERNEL_WIDTH} \u00d7 {processing.KERNEL_WIDTH} pixels "
            "around. Keep the change?",
            before,
            after,
            mask,
            self.get_scaling_fn(),
            self.parent.diffraction_space_widget.ui.histogram.gradient.colorMap(),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            processing.restore_pixels(datacube, mask, original)
            self.pixels_status.setText(f"Undid replacing {n} {kind} pixel{plural}.")
            return

        self.pixels_status.setText(f"Replaced {n} {kind} pixel{plural}.")
        self.patterns = {"mean": after}  # the maximum pattern has changed too
        self._announce_data_changed()

    def _announce_data_changed(self):
        # Tell the main window (and other plugins) the dataset's values have
        # changed in place -- without the plugin rebuilding this window, as
        # its apertures and previews still apply -- then redraw.
        self.keep_through_data_change = True
        try:
            self.parent.set_datacube(self.parent.datacube, self.parent.windowTitle())
        finally:
            self.keep_through_data_change = False
        self.update_vis()
        self.update_dps()
        if self.merge_button.isChecked():
            self._show_merged()
