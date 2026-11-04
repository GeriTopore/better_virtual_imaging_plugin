import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
)
from skimage.filters import threshold_otsu

from .save_dialog import SaveDialog

BLACK_WHITE = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
MASK_BACKGROUND = (110, 110, 110)


def default_threshold(image):
    """Otsu's threshold, splitting the image's intensities in two."""
    finite = image[np.isfinite(image)]
    if finite.size == 0 or finite.min() == finite.max():
        return float(finite.mean()) if finite.size else 0.0
    return float(threshold_otsu(finite))


def threshold_mask(image, threshold, invert=False):
    """The pixels at or above the threshold -- or, inverted, those below."""
    mask = image >= threshold
    return ~mask if invert else mask


class MaskDialog(SaveDialog):
    """
    Saving a virtual image as a mask: the scan positions whose intensity is
    at or above a threshold (white), or with Invert those below it. Shown
    beside the virtual image, zooming and panning together; saved as a
    boolean NumPy array indexed [rx, ry].
    """

    file_filters = "NumPy array (*.npy)"
    default_extension = ".npy"
    format_examples = ".npy"

    def __init__(
        self, image, cmap, default_path, threshold=None, invert=False, parent=None
    ):
        super().__init__("Save Mask", default_path, parent=parent)
        self.image = np.asarray(image, dtype=float)
        self.mask = None

        # Displayed transposed, like the previews.
        views = []
        for shown, colormap, levels in (
            (self.image.T, cmap, None),
            (np.zeros(self.image.T.shape), BLACK_WHITE, (0, 1)),
        ):
            view = pg.ImageView()
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.ui.histogram.hide()
            view.setColorMap(colormap)
            if levels is None:
                view.setImage(shown)
            else:
                view.setImage(shown, autoLevels=False, levels=levels)
            views.append(view)
        self.image_view, self.mask_view = views
        # Grey around the mask, so black (not in it) shows up to its edges.
        self.mask_view.ui.graphicsView.setBackground(MASK_BACKGROUND)
        self.mask_view.getView().setXLink(self.image_view.getView())
        self.mask_view.getView().setYLink(self.image_view.getView())

        finite = self.image[np.isfinite(self.image)]
        low, high = (finite.min(), finite.max()) if finite.size else (0.0, 0.0)
        if threshold is None:
            threshold = default_threshold(self.image)

        self.threshold_edit = QLineEdit(f"{threshold:.6g}")
        self.threshold_edit.setToolTip(
            "Scan positions at or above this intensity are in the mask (white)"
        )
        self.threshold_edit.textChanged.connect(self.update_mask)
        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.setToolTip(
            "Make the mask the scan positions below the threshold instead"
        )
        self.invert_checkbox.setChecked(invert)
        self.invert_checkbox.toggled.connect(self.update_mask)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Intensity threshold"))
        controls.addWidget(self.threshold_edit)
        controls.addWidget(QLabel(f"(the image ranges {low:.6g} to {high:.6g})"))
        controls.addSpacing(20)
        controls.addWidget(self.invert_checkbox)
        controls.addStretch()

        views_layout = QGridLayout()
        views_layout.addWidget(QLabel("Virtual image"), 0, 0)
        views_layout.addWidget(QLabel("Mask"), 0, 1)
        views_layout.addWidget(self.image_view, 1, 0)
        views_layout.addWidget(self.mask_view, 1, 1)
        views_layout.setColumnStretch(0, 1)
        views_layout.setColumnStretch(1, 1)
        views_layout.setRowStretch(1, 1)

        self.content_layout.addLayout(controls)
        self.content_layout.addLayout(views_layout, stretch=1)

        self.update_mask()
        self.resize(1000, 650)

    def threshold(self):
        """The threshold typed, or None if it isn't a number."""
        try:
            value = float(self.threshold_edit.text())
        except ValueError:
            return None
        return value if np.isfinite(value) else None

    def invert(self):
        return self.invert_checkbox.isChecked()

    def update_mask(self, *_):
        threshold = self.threshold()
        # Nothing to save until the threshold is a number.
        self.save_button.setEnabled(threshold is not None)
        if threshold is None:
            return
        self.mask = threshold_mask(self.image, threshold, self.invert())
        self.mask_view.setImage(
            self.mask.T.astype(float), autoLevels=False, levels=(0, 1), autoRange=False
        )

    def formats(self):
        return {"npy"}

    def write(self, path):
        np.save(path, self.mask)
