import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

MARKER_PEN = {"color": "r", "width": 2}


class PixelCorrectionDialog(QDialog):
    """
    The mean pattern before and after replacing bad pixels, with the pixels
    found ringed on both, for the user to keep (OK) or undo the change. The
    two views zoom and pan together.
    """

    def __init__(
        self, title, summary, before, after, mask, scale_fn, cmap, parent=None
    ):
        super().__init__(parent=parent)
        self.setWindowTitle(title)

        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)

        # Both on the levels of the corrected pattern: hot pixels would
        # otherwise stretch the uncorrected one's range out of sight.
        scaled_before, scaled_after = scale_fn(before).T, scale_fn(after).T
        levels = tuple(np.percentile(scaled_after, [0.1, 99.9]))

        # Displayed transposed, with data pixel (qx, qy) centred at
        # (qy + 0.5, qx + 0.5).
        qx, qy = np.nonzero(mask)
        views = []
        for image in (scaled_before, scaled_after):
            view = pg.ImageView()
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.setColorMap(cmap)
            view.setImage(image, autoLevels=False, levels=levels)
            markers = pg.ScatterPlotItem(
                x=qy + 0.5,
                y=qx + 0.5,
                size=14,
                symbol="o",
                pen=pg.mkPen(**MARKER_PEN),
                brush=None,
            )
            view.getView().addItem(markers)
            views.append(view)
        self.before_view, self.after_view = views
        self.after_view.getView().setXLink(self.before_view.getView())
        self.after_view.getView().setYLink(self.before_view.getView())

        views_layout = QGridLayout()
        views_layout.addWidget(QLabel("Before: mean pattern"), 0, 0)
        views_layout.addWidget(QLabel("After: mean pattern"), 0, 1)
        views_layout.addWidget(self.before_view, 1, 0)
        views_layout.addWidget(self.after_view, 1, 1)
        views_layout.setColumnStretch(0, 1)
        views_layout.setColumnStretch(1, 1)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Put the original pixel values back")
        self.undo_button.clicked.connect(self.reject)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        # See BetterVirtualImagingWindow.disable_auto_default.
        for button in (self.undo_button, self.ok_button):
            button.setAutoDefault(False)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.undo_button)
        buttons.addWidget(self.ok_button)

        layout = QVBoxLayout(self)
        layout.addWidget(summary_label)
        layout.addLayout(views_layout, stretch=1)
        layout.addLayout(buttons)

        self.resize(1000, 600)
