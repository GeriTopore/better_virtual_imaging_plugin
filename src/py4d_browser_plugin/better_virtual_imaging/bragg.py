"""
Finding and reading Bragg vectors saved alongside the loaded dataset (e.g.
by the Disk Detection plugin, which appends them to the file the raw data
came from).
"""

import os

import h5py
import numpy as np
import py4DSTEM

from py4D_browser.menu_actions import get_ND

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from py4D_browser import DataViewer

# The extensions py4D_browser.menu_actions.load_file reads as HDF5.
H5_EXTENSIONS = (".h5", ".hdf5", ".py4dstem", ".emd", ".mat")


def loaded_filepath(parent: "DataViewer") -> Optional[str]:
    """
    The file the current dataset was loaded from, if known. py4D_browser
    doesn't record this anywhere other than the window title, which
    load_file sets to the path (and set_datacube, used by other plugins, to
    something else).
    """
    title = parent.windowTitle()
    return os.path.abspath(title) if os.path.isfile(title) else None


def _shapes_match(braggvectors_group, data_shape) -> bool:
    try:
        shape = braggvectors_group["metadatabundle/_braggvectors_shape"]
        Rshape, Qshape = tuple(shape["Rshape"][()]), tuple(shape["Qshape"][()])
    except KeyError:
        return False
    return Rshape + Qshape == tuple(data_shape)


def find_braggvectors(parent: "DataViewer") -> Optional[tuple[str, str]]:
    """
    Find (filepath, datapath) of Bragg vectors saved in the file the current
    dataset was loaded from, for the same scan and pattern shapes. Checks
    the file's first 4D dataset (the one load_file reads) is the same shape
    as the loaded data, in case the window title is stale, and prefers
    vectors saved under that dataset's own group.
    """
    filepath = loaded_filepath(parent)
    if filepath is None or os.path.splitext(filepath)[-1].lower() not in H5_EXTENSIONS:
        return None

    data_shape = parent.datacube.data.shape
    found = []

    def visit(name, obj):
        python_class = obj.attrs.get("python_class")
        if python_class in ("BraggVectors", b"BraggVectors") and _shapes_match(
            obj, data_shape
        ):
            found.append("/" + name)

    try:
        with h5py.File(filepath, "r") as file:
            datacubes = get_ND(file)
            if len(datacubes) == 0 or datacubes[0].shape != data_shape:
                return None
            group = "/".join(datacubes[0].name.split("/")[:-1])
            file.visititems(visit)
    except OSError:
        return None

    if not found:
        return None
    found.sort(key=lambda datapath: not datapath.startswith(group + "/"))
    return filepath, found[0]


def read_braggvectors(filepath: str, datapath: str):
    return py4DSTEM.read(filepath, datapath=datapath)


def bvm(braggvectors) -> np.ndarray:
    """
    The Bragg vector map, from the vectors' raw (detector pixel) positions,
    so it lines up with the diffraction patterns.
    """
    return np.asarray(braggvectors.histogram(mode="raw").data)


def peaks_at(braggvectors, positions) -> np.ndarray:
    """
    The raw (qx, qy) of every Bragg peak detected at the given scan
    positions, as an (N, 2) array.
    """
    peaks = [braggvectors.raw[rx, ry] for rx, ry in positions]
    if not peaks:
        return np.zeros((0, 2))
    return np.column_stack(
        [np.concatenate([p.qx for p in peaks]), np.concatenate([p.qy for p in peaks])]
    )
