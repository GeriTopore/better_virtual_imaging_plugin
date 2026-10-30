# CLAUDE.md

An external plugin for [py4D-browser](https://github.com/py4dstem/py4D-browser) (a PyQt5 GUI for 4D-STEM data built on py4DSTEM) for better virtual imaging. Sister project of `~/Documents/Dev/disk_detection_plugin`, which follows the same rules below and is the best reference for working patterns.

py4D-browser's own architecture notes (local, untracked file -- skipped if absent):
@~/Documents/Dev/py4D-browser/CLAUDE.md

## Hard rule: never modify py4D-browser
The plugin must work against stock py4D-browser, installed separately. Work around host or py4DSTEM limitations inside the plugin.
- Before relying on any host API, check it exists in the user's fork's synced `dev` branch (`~/Documents/Dev/py4D-browser`, remote `origin` = GeriTopore/py4D-browser, synced with upstream py4dstem/py4D-browser) -- not on some local feature branch.
- Host APIs known to work: `parent.datacube`, `parent.statusBar()`, `parent.qtapp`, `parent.get_virtual_image_detector()` (DetectorInfo: shape POINT -> `point`, RECTANGULAR -> `slice`/`mask`), `parent.unscaled_realspace_image`, `parent.unscaled_diffraction_image`, `parent.real_space_widget` / `parent.diffraction_space_widget` (pyqtgraph ImageViews), `parent.show_file_dialog()`, `parent.get_savefile_name(fmt)`, `parent.signal_datacube_changed`, `parent.signal_diffraction_data_changed` (fires on every main DP redraw, including real-space detector shape/response changes), `py4D_browser.utils` (`DetectorShape`, `DetectorMode`, `DetectorInfo`, `pg_point_roi`), `py4D_browser.menu_actions` (`get_ND`, `find_calibrations`), `py4D_browser.dialogs.ResizeDialog`.

## Plugin mechanics
- Package lives in the `py4d_browser_plugin` namespace package: `src/py4d_browser_plugin/better_virtual_imaging/`. There must be **no** `__init__.py` at the `py4d_browser_plugin` level.
- The host's `load_plugins` imports every module in that namespace and instantiates any class with a `plugin_id` as `Cls(parent=DataViewer, plugin_menu=..., plugin_action=...)`. `uses_single_action = True` gets a menu entry under Plugins, passed in as `plugin_action`.
- The host calls the plugin's `close()` on shutdown (since 1.5.3).
- `parent.signal_datacube_changed` fires on every new dataset (File > Open and other plugins' `set_datacube`) -- the plugin resets its window on it.
- The loaded file's path is only recorded as the main window title (set by `load_file`); validate it (`os.path.isfile`, matching data shape) before trusting it.
- `DataViewer()` takes `filepath=`, `reset_state=`, `debug_console=` -- not `argv` (changed in 1.5.3).

## Known quirks
- **Display transposition:** the host displays images transposed (`image.T`) in its ImageViews; mirror that, and read point-selector positions with the host's convention: `y0, x0 = roi.saveState()["pos"]; rx, ry = int(x0 + 1), int(y0 + 1)`. Scatter markers over a transposed image need `(y, x)` positions.
- **QDialog buttons:** every QPushButton in a QDialog is "auto-default" -- the first one renders highlighted and Enter in any spinbox clicks it. Call `setAutoDefault(False)` on all buttons, including ones added later.
- **Colormaps:** copy them from the main window's ImageViews (`widget.ui.histogram.gradient.colorMap()`) and follow `sigGradientChangeFinished`, rather than re-reading settings.
- **py4DSTEM 0.14.18 / emdfile 0.0.16:**
  - `DataCube.get_vacuum_probe(ROI=(rx_min, rx_max, ry_min, ry_max))` uses the wrong scan positions (indices relative to the ROI's corner) -- pass a boolean mask.
  - `get_vacuum_probe` fails for a single-position ROI.
  - `py4DSTEM.save(..., mode="a", emdpath=...)` silently writes nothing if the in-memory root name already exists in the file, and can't overwrite existing nodes; always verify writes.
  - numpy scalars repr as `np.float64(...)` -- convert to Python types before writing text meant to be parsed back.

## Workflow
- Format with `black` (`black src`); there is no test suite.
- Dev environment: conda env `4dbrowser-dev`, with py4D-browser and this plugin installed editable (`pip install -e .`); relaunch `py4DGUI` to pick up changes.
- Test headlessly by constructing `DataViewer()` with `QT_QPA_PLATFORM=offscreen` (or `cocoa` for native macOS rendering/screenshots), finding the plugin in `w.loaded_plugins`, and driving it directly; patch modal dialogs (`QMessageBox.exec_`, `.question`) so they don't block.
- Test data: use scratch copies of `.h5` files and never write to the user's originals. `.h5` and other data formats are git-ignored.
- Commit on a feature branch, then fast-forward `main`; push only when asked.
