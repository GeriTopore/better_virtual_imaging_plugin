# better_virtual_imaging_plugin
A plugin for py4d_browser for better virtual imaging of 4D-STEM data.

## Installation
This is an external plugin for [py4D-browser](https://github.com/py4dstem/py4D-browser) -- install py4D-browser first, then install this on top of it:
```bash
pip install git+https://github.com/GeriTopore/better_virtual_imaging_plugin
```
The plugin is discovered automatically the next time you launch `py4DGUI`, and appears as "Better Virtual Imaging..." under the Plugins menu.

## Usage
Plugins > Better Virtual Imaging... opens a window laid out like the main one: in each preview, a diffraction pattern on the left and a virtual image on the right. The settings are in the column on the left.

- **Virtual apertures:** turn on "Add Aperture on Click", then click a diffraction pattern to place a circular aperture of the chosen radius there. Drag an aperture to move it, use its handle to resize it, and right-click it to remove it. "Use BF radius" sizes apertures to the bright-field disk, measured from the mean diffraction pattern. The virtual image is the sum over all of a preview's apertures. Until a preview has any, it shows the main window's virtual image to navigate by. "Clear Apertures" removes them all.
- **g-vector arrays:** "g-Vector Array" puts apertures on the lattice spanned by two g-vectors. Drag the markers into place: O (starting at the bright-field disk's centre) and the tips of g1 and g2. Moving O carries both vectors with it. For each vector, set how many repeats the array extends either way along it, and a shift along it (e.g. 0.5 for half-order spots). "Include origin" adds an aperture on the direct beam. The array's apertures follow the radius setting and add to any click-placed ones.
- **Real-space cursor:** move it over the virtual image to choose which diffraction pattern is shown. It follows the main window's Detector Shape > Virtual Image setting: a point shows one pattern, and a rectangle shows the mean over it (or the maximum, with Detector Response > Virtual Image > Maximum).
- **Previews:** "Add Preview" adds up to three more, each with its own apertures, array and cursor, so different aperture sets can be compared side by side. The x on a preview removes it.
- **Merged view:** "Merge Views" combines the previews with apertures into one view: their virtual images as a colour composite, and all their apertures on one diffraction pattern, each preview in its own colour. Click a colour in the key under the composite to change it. Click "Merge Views" again to go back to the separate previews.
- **Diffraction display:** Linear, Log (the default) or Power scaling for the diffraction patterns and Bragg vector map. Colormaps follow the main window's.

### With Bragg vectors
If the file the dataset was loaded from also has Bragg vectors saved in it for the same data (for example by the Disk Detection plugin), the plugin also offers:
- **Bragg vector map:** shown below the settings.
- **Show detected peaks:** marks the Bragg peaks detected at each preview's cursor with red dots -- every scan position inside a rectangular cursor.
- **Snap to Bragg peaks:** for a g-vector array, puts O, g1 and g2 on the peaks detected in the pattern shown -- all three when it's ticked or the array is first switched on, and each one when dropped near a peak.
