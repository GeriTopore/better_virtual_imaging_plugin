import os

from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from . import figure

# Offered when browsing; any other format matplotlib saves also works.
FILE_FILTERS = "PNG (*.png);;TIFF (*.tif *.tiff);;PDF (*.pdf);;SVG (*.svg)"
DEFAULT_EXTENSION = ".png"

# Width of the preview of the figure, in screen pixels.
PREVIEW_WIDTH = 900


class SaveFigureDialog(QDialog):
    """
    A preview of a figure about to be saved, and where to: a path to edit,
    or choose with Browse. The format follows the file's extension. After
    exec_(), `saved_path` is where it was saved, if it was.
    """

    def __init__(self, fig, default_path, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Save Image")

        self.fig = fig
        self.saved_path = None
        # A path Browse already confirmed overwriting, so not asked again.
        self._confirmed_path = None

        preview = QLabel()
        preview.setAlignment(QtCore.Qt.AlignCenter)
        pixmap = QPixmap()
        pixmap.loadFromData(
            figure.render_png(fig, dpi=PREVIEW_WIDTH / fig.get_figwidth())
        )
        preview.setPixmap(pixmap)

        self.path_edit = QLineEdit(default_path)
        self.path_edit.setMinimumWidth(500)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Save to"))
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)

        # See BetterVirtualImagingWindow.disable_auto_default.
        for button in (browse_button, cancel_button, save_button):
            button.setAutoDefault(False)

        layout = QVBoxLayout(self)
        layout.addWidget(preview, stretch=1)
        layout.addLayout(path_row)
        layout.addLayout(buttons)

    def _path(self):
        path = os.path.abspath(os.path.expanduser(self.path_edit.text().strip()))
        if not os.path.splitext(path)[1]:
            path += DEFAULT_EXTENSION
        return path

    def browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", self._path(), FILE_FILTERS
        )
        if path:
            self.path_edit.setText(path)
            # The file dialog has asked about overwriting it.
            self._confirmed_path = self._path()

    def save(self):
        if not self.path_edit.text().strip():
            return
        path = self._path()
        extension = os.path.splitext(path)[1][1:].lower()
        if extension not in figure.supported_formats(self.fig):
            QMessageBox.warning(
                self,
                "Save Image",
                f"Can't save as .{extension}. Use, for example, .png, .tif, "
                ".pdf or .svg.",
            )
            return
        if not os.path.isdir(os.path.dirname(path)):
            QMessageBox.warning(
                self, "Save Image", f"There's no folder {os.path.dirname(path)}."
            )
            return
        if os.path.exists(path) and path != self._confirmed_path:
            answer = QMessageBox.question(
                self,
                "Save Image",
                f"{os.path.basename(path)} already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            self.fig.savefig(path, dpi=figure.SAVE_DPI)
        except Exception as exc:
            QMessageBox.warning(self, "Save Image", f"Couldn't save it: {exc}")
            return
        self.saved_path = path
        self.accept()
