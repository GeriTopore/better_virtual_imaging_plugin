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

# Width of the preview of a figure, in screen pixels.
PREVIEW_WIDTH = 900


class SaveDialog(QDialog):
    """
    Saving something to a file: whatever subclasses show above a path to
    edit, or choose with Browse, and Cancel and Save. The format follows the
    file's extension. After exec_(), `saved_path` is where it was saved, if
    it was. Subclasses add their content to `content_layout`, and give
    `formats()` and `write(path)`.
    """

    # For the Browse dialog, and to add to a path given without an extension.
    file_filters = ""
    default_extension = ".png"
    # For telling the user which extensions to use.
    format_examples = ".png"

    def __init__(self, title, default_path, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(title)

        self.saved_path = None
        # A path Browse already confirmed overwriting, so not asked again.
        self._confirmed_path = None

        self.content_layout = QVBoxLayout()

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
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(self.save_button)

        # See BetterVirtualImagingWindow.disable_auto_default.
        for button in (browse_button, cancel_button, self.save_button):
            button.setAutoDefault(False)

        layout = QVBoxLayout(self)
        layout.addLayout(self.content_layout, stretch=1)
        layout.addLayout(path_row)
        layout.addLayout(buttons)

    def formats(self):
        """The file extensions it can be saved as, e.g. "png"."""
        raise NotImplementedError

    def write(self, path):
        raise NotImplementedError

    def _path(self):
        path = os.path.abspath(os.path.expanduser(self.path_edit.text().strip()))
        if not os.path.splitext(path)[1]:
            path += self.default_extension
        return path

    def browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, self.windowTitle(), self._path(), self.file_filters
        )
        if path:
            self.path_edit.setText(path)
            # The file dialog has asked about overwriting it.
            self._confirmed_path = self._path()

    def save(self):
        if not self.path_edit.text().strip():
            return
        title = self.windowTitle()
        path = self._path()
        extension = os.path.splitext(path)[1][1:].lower()
        if extension not in self.formats():
            if len(self.formats()) == 1:
                hint = f"Save it as {self.format_examples}."
            else:
                hint = f"Use, for example, {self.format_examples}."
            QMessageBox.warning(self, title, f"Can't save as .{extension}. {hint}")
            return
        if not os.path.isdir(os.path.dirname(path)):
            QMessageBox.warning(
                self, title, f"There's no folder {os.path.dirname(path)}."
            )
            return
        if os.path.exists(path) and path != self._confirmed_path:
            answer = QMessageBox.question(
                self,
                title,
                f"{os.path.basename(path)} already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            self.write(path)
        except Exception as exc:
            QMessageBox.warning(self, title, f"Couldn't save it: {exc}")
            return
        self.saved_path = path
        self.accept()


class SaveFigureDialog(SaveDialog):
    """A preview of a figure about to be saved, and where to."""

    file_filters = "PNG (*.png);;TIFF (*.tif *.tiff);;PDF (*.pdf);;SVG (*.svg)"
    format_examples = ".png, .tif, .pdf or .svg"

    def __init__(self, fig, default_path, parent=None):
        super().__init__("Save Image", default_path, parent=parent)
        self.fig = fig

        preview = QLabel()
        preview.setAlignment(QtCore.Qt.AlignCenter)
        pixmap = QPixmap()
        pixmap.loadFromData(
            figure.render_png(fig, dpi=PREVIEW_WIDTH / fig.get_figwidth())
        )
        preview.setPixmap(pixmap)
        self.content_layout.addWidget(preview)

    def formats(self):
        # Any matplotlib saves.
        return figure.supported_formats(self.fig)

    def write(self, path):
        self.fig.savefig(path, dpi=figure.SAVE_DPI)
