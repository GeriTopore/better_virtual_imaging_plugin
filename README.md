# better_virtual_imaging_plugin
A plugin for py4d_browser for better virtual imaging of 4D-STEM data.

## Installation
This is an external plugin for [py4D-browser](https://github.com/py4dstem/py4D-browser) -- install py4D-browser first, then install this on top of it:
```bash
pip install git+https://github.com/GeriTopore/better_virtual_imaging_plugin
```
The plugin is discovered automatically the next time you launch `py4DGUI`, and appears as "Better Virtual Imaging..." under the Plugins menu.

## Usage
Plugins > Better Virtual Imaging... opens a window laid out like the main one: a diffraction pattern on the left and a virtual image on the right.

- **Virtual apertures:** turn on "Add Aperture on Click", then click a diffraction pattern to place a circular aperture of the chosen radius there. Drag an aperture to move it, use its handle to resize it, and right-click it to remove it. The virtual image is the sum over all of a preview's apertures. Until a preview has any, it shows the main window's virtual image to navigate by.
- **Real-space cursor:** move it over the virtual image to choose which diffraction pattern is shown. It follows the main window's Detector Shape > Virtual Image setting: a point shows one pattern, and a rectangle shows the mean over it (or the maximum, with Detector Response > Virtual Image > Maximum).
- **More previews:** "Add Preview" adds up to three more, each with its own apertures and cursor, so different aperture sets can be compared side by side.
