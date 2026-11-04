"""
Scrolling through the previews with the mouse wheel or trackpad, wherever
the pointer is: over a preview's images (or their histograms), the wheel
scrolls the previews, and zooms the image only with Cmd (macOS) or Ctrl
held -- as in maps embedded in web pages. Spin boxes and menus in the
previews likewise only take the wheel once clicked into.
"""

import sys
import time

from PyQt5 import QtCore
from PyQt5.QtWidgets import QAbstractSpinBox, QApplication, QComboBox

# Qt reports Cmd on macOS as Control.
ZOOM_KEY = "⌘" if sys.platform == "darwin" else "Ctrl"

# A mouse wheel's clicks, unlike a trackpad gesture, don't say when a scroll
# starts or ends: clicks this close together (in seconds) count as one
# scroll, which zooms or scrolls throughout, whatever keys are let go or
# pressed partway.
WHEEL_LATCH = 0.4


class WheelRouter(QtCore.QObject):
    """Routes the wheel over previews' widgets to the scroll area they're in."""

    def __init__(self, scroll_area, parent=None):
        super().__init__(parent)
        self.scroll_area = scroll_area
        self.views = set()  # image and histogram viewports, which zoom
        self._zooming = None  # whether the current scroll zooms
        self._last_event = 0.0

    def add_pane(self, pane):
        """Route the wheel over a preview's images, histograms and fields."""
        for image_view in (pane.dp_view, pane.vi_view):
            for widget in (image_view.ui.graphicsView, image_view.ui.histogram):
                self.views.add(widget.viewport())
                widget.viewport().installEventFilter(self)
        for field in pane.findChildren((QAbstractSpinBox, QComboBox)):
            # Only take focus by being clicked into (or tabbed to), not
            # from the wheel passing over.
            field.setFocusPolicy(QtCore.Qt.StrongFocus)
            field.installEventFilter(self)

    def remove_pane(self, pane):
        for image_view in (pane.dp_view, pane.vi_view):
            for widget in (image_view.ui.graphicsView, image_view.ui.histogram):
                self.views.discard(widget.viewport())

    def eventFilter(self, obj, event):
        if event.type() != QtCore.QEvent.Wheel:
            return False
        if obj in self.views:
            if self._zooms(event):
                return False  # the view zooms, as it would anyway
        elif obj.hasFocus():
            return False  # a field clicked into takes the wheel
        # Scroll the previews, as the scroll area does with the wheel.
        QApplication.sendEvent(self.scroll_area.verticalScrollBar(), event)
        return True

    def _zooms(self, event):
        # Decided when a scroll starts, from whether Cmd/Ctrl is held then.
        now = time.monotonic()
        phase = event.phase()
        continuing = self._zooming is not None and (
            phase
            in (QtCore.Qt.ScrollUpdate, QtCore.Qt.ScrollMomentum, QtCore.Qt.ScrollEnd)
            or (
                phase == QtCore.Qt.NoScrollPhase
                and now - self._last_event < WHEEL_LATCH
            )
        )
        if not continuing:
            self._zooming = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        self._last_event = now
        return self._zooming
