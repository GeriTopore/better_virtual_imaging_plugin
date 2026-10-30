import os
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
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.utils import DetectorMode, DetectorShape

from . import bragg, imaging
from .preview_pane import BVMPane, MergedPane, PreviewPane

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


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
            self.window.load_bvm()

        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def on_datacube_changed(self):
        # Anything the window computed belongs to the old dataset, so start
        # over with a fresh window -- reopened in the same place if it was
        # open.
        if self.window is None:
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
    # Start with a single preview pane; this caps how many more the user can
    # add via the "Add Preview" button.
    MAX_ADDITIONAL_PANES = 3

    def __init__(self, parent: "DataViewer"):
        super().__init__(parent=parent)

        self.parent = parent
        self.panes = []

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

        apertures_box = self._build_apertures_box()
        scaling_box = self._build_scaling_box()

        self.add_pane_button = QPushButton("Add Preview")
        self.add_pane_button.clicked.connect(self.add_pane)

        # Toggles between the separate previews and a merged view of them
        # (built on first use, then hidden rather than deleted).
        self.merged_pane = None
        self.merged = []  # the previews in the merged view, in order
        self.merge_button = QPushButton("Merge Views")
        self.merge_button.setCheckable(True)
        self.merge_button.toggled.connect(self.on_merge_toggled)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(apertures_box)
        controls_layout.addWidget(scaling_box)
        controls_layout.addWidget(self.add_pane_button)
        controls_layout.addWidget(self.merge_button)
        controls_layout.addStretch()
        self.controls_widget = QWidget()
        self.controls_widget.setLayout(controls_layout)

        # At most half the window's height (see _fit_controls_height),
        # scrolling vertically beyond that -- and never squeezed narrower
        # than the controls need.
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(self.controls_widget)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(
            self.controls_widget.minimumSizeHint().width()
            + self.controls_scroll.verticalScrollBar().sizeHint().width()
        )

        # The left column: the controls, and below them the Bragg vector map
        # if there are Bragg vectors saved with the dataset.
        self.bvm_pane = None
        self.left_layout = QVBoxLayout()
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.addWidget(self.controls_scroll)
        self.left_layout.addStretch()
        left_column = QWidget()
        left_column.setLayout(self.left_layout)

        self.previews_layout = QGridLayout()
        self.views = QWidget()
        self.views.setLayout(self.previews_layout)

        self.main_splitter = QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(left_column)
        self.main_splitter.addWidget(self.views)
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
        self.load_bvm()
        self._fit_controls_height()

        self.disable_auto_default(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_controls_height()

    def _fit_controls_height(self):
        content_height = (
            self.controls_widget.sizeHint().height()
            + 2 * self.controls_scroll.frameWidth()
        )
        self.controls_scroll.setFixedHeight(min(content_height, self.height() // 2))

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

        # Locks new apertures to the bright-field disk radius, measured from
        # the mean diffraction pattern the first time it's checked.
        self.bf_radius = None
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
        scaling_box = QGroupBox("Diffraction Scaling")
        scaling_layout = QVBoxLayout()

        self.scaling_group = QButtonGroup(self)
        self.linear_radio = QRadioButton("Linear")
        self.log_radio = QRadioButton("Log")
        self.power_radio = QRadioButton("Power")
        self.power_radio.setChecked(True)
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

        scaling_box.setLayout(scaling_layout)
        return scaling_box

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
        if len(self.panes) - 1 >= self.MAX_ADDITIONAL_PANES:
            return

        pane = PreviewPane(f"Preview {len(self.panes) + 1}")
        pane.sig_scan_selection_changed.connect(partial(self.update_dps, panes=[pane]))
        pane.sig_apertures_changed.connect(partial(self.update_vis, panes=[pane]))
        pane.sig_dp_clicked.connect(partial(self.on_dp_clicked, pane))
        pane.set_click_to_add(self.add_on_click_button.isChecked())
        self.panes.append(pane)
        self._layout_panes()
        self.disable_auto_default(pane)
        self.sync_colormaps(panes=[pane])

        self._place_cursor_from_main(pane)
        self.update_vis(panes=[pane])
        self.update_dps(panes=[pane])

        if len(self.panes) - 1 >= self.MAX_ADDITIONAL_PANES:
            self.add_pane_button.setEnabled(False)

    def add_pane(self):
        self._append_pane()

    def _layout_panes(self):
        # One pane fills the window, like the main window; two stack, and
        # more go in a 2x2 grid.
        columns = 1 if len(self.panes) <= 2 else 2
        for pane in self.panes:
            self.previews_layout.removeWidget(pane)
        for i, pane in enumerate(self.panes):
            self.previews_layout.addWidget(pane, i // columns, i % columns)

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
        if self.bvm_pane is not None:
            self.bvm_pane.set_colormap(diffraction_cmap)

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
            return self.bf_radius
        return self.radius_spin.value()

    def on_use_bf_radius_toggled(self, checked):
        if checked and self.bf_radius is None:
            self.bf_radius = self._measure_bf_radius()
            if self.bf_radius is None:
                # Leave the radius under the user's control.
                self.use_bf_radius_checkbox.setChecked(False)
                return
        if checked:
            self.radius_spin.setValue(self.bf_radius)
        self.radius_spin.setEnabled(not checked)

    def _measure_bf_radius(self):
        if not self._has_data():
            return None
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return imaging.measure_bf_radius(self.parent.datacube.data)
        except Exception as exc:
            self.parent.statusBar().showMessage(
                f"Couldn't measure the BF disk radius: {exc}", 5_000
            )
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def on_scaling_changed(self, *_):
        # Redisplay each pane's cached pattern with fresh levels for the new
        # scaling.
        scale_fn = self.get_scaling_fn()
        for pane in self._cursor_panes():
            if pane.last_dp is not None:
                pane.show_dp(pane.last_dp, scale_fn, relevel=True)
        if self.bvm_pane is not None:
            self.bvm_pane.show_bvm(scale_fn)

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
        merged = [pane for pane in self.panes if pane.vimg is not None]
        if not merged or not self._has_data():
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
            self.sync_colormaps(panes=[self.merged_pane])

        # Start the cursor where the first merged preview's is.
        Rshape = self.data_shape[:2]
        source = merged[0]
        xc, yc = source.scan_centre(Rshape)
        if source.rs_shape is DetectorShape.POINT:
            self.merged_pane.set_point_selector(xc, yc)
        else:
            like = source.get_scan_selection(Rshape)
            self.merged_pane.set_rect_selector(*self._rect_around(xc, yc, like=like))

        self.merged = merged
        self._show_merged()

        for pane in self.panes:
            self.previews_layout.removeWidget(pane)
            pane.hide()
        self.previews_layout.addWidget(self.merged_pane, 0, 0)
        self.merged_pane.show()
        self.update_dps(panes=[self.merged_pane])

        self.add_pane_button.setEnabled(False)
        return True

    def _merged_colors(self):
        # Each merged preview's chosen colour, or else its default.
        defaults = imaging.composite_colors(len(self.merged))
        return [
            pane.color if pane.color is not None else default
            for pane, default in zip(self.merged, defaults)
        ]

    def _show_merged(self):
        colors = self._merged_colors()
        self.merged_pane.show_composite(
            imaging.color_composite([pane.vimg for pane in self.merged], colors)
        )
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
        self.previews_layout.removeWidget(self.merged_pane)
        self.merged_pane.hide()
        for pane in self.panes:
            pane.show()
        self._layout_panes()
        self.add_pane_button.setEnabled(len(self.panes) - 1 < self.MAX_ADDITIONAL_PANES)

    # --- Bragg vector map ---

    def load_bvm(self):
        # Show the Bragg vector map of any Bragg vectors saved with the
        # dataset. Checked again whenever the window is reopened, as they may
        # have been saved since (e.g. by the Disk Detection plugin).
        if self.bvm_pane is not None or not self._has_data():
            return
        source = bragg.find_braggvectors(self.parent)
        if source is None:
            return
        filepath, datapath = source

        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            braggvectors = bragg.read_braggvectors(filepath, datapath)
            bvm = bragg.bvm(braggvectors)
        except Exception as exc:
            self.parent.statusBar().showMessage(
                f"Couldn't read the Bragg vectors at {datapath}: {exc}", 5_000
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.bvm_pane = BVMPane(bvm, f"{os.path.basename(filepath)}: {datapath}")
        # Fills the column below the controls, in place of the stretch.
        self.left_layout.insertWidget(1, self.bvm_pane, stretch=1)
        self.sync_colormaps(panes=[])
        self.bvm_pane.show_bvm(self.get_scaling_fn(), first=True)

        # Widen the column to give the map room. (Until the window is first
        # shown, the splitter reports placeholder sizes, which it later
        # rescales in proportion.)
        if not self.isVisible():
            self.main_splitter.setSizes([400, self.width() - 400])
        elif self.main_splitter.sizes()[0] < 400:
            total = sum(self.main_splitter.sizes())
            self.main_splitter.setSizes([400, total - 400])
