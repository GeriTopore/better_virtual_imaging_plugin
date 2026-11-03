"""
Processing the dataset in place: binning, with py4DSTEM's bin_Q and bin_R,
and replacing hot or dead pixels -- pixels that are bad in every pattern,
and so stand out in the mean pattern -- with py4DSTEM's
median_filter_masked_pixels.
"""

import contextlib
import io

import h5py
import numpy as np
import py4DSTEM
from py4DSTEM.preprocess import filter_hot_pixels
from scipy.ndimage import median_filter

# The neighbourhood each bad pixel's replacement is the median of (as in
# py4DSTEM's filter_hot_pixels).
KERNEL_WIDTH = 3


@contextlib.contextmanager
def _quiet():
    # py4DSTEM prints progress bars and notes for what are, here, small jobs.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yield


def can_bin(data):
    # py4DSTEM bins by reshaping, which an HDF5 dataset (memory-mapped
    # loading of an HDF5 file) can't do. A numpy memmap can, into memory.
    return not isinstance(data, h5py.Dataset)


def can_modify(data):
    # Changing memory-mapped data would write into the file it's mapped from
    # (or fail, if that's read-only).
    return isinstance(data, np.ndarray) and not isinstance(data, np.memmap)


def binned_dtype(data, N):
    """
    A dtype that holds the sums of N x N pixels: the data's own, unless
    they're integers that could overflow it. (bin_Q would otherwise wrap
    them around in the data's dtype, and bin_R widen them to 64 bits.)
    """
    dtype = data.dtype
    if not np.issubdtype(dtype, np.integer):
        return dtype
    peak = int(np.max(data)) * N * N
    if peak <= np.iinfo(dtype).max:
        return dtype
    return np.promote_types(dtype, np.min_scalar_type(peak))


def bin_datacube(datacube, axis, N):
    """Bin the datacube in place by N, in diffraction ("Q") or real ("R") space."""
    dtype = binned_dtype(datacube.data, N)
    with _quiet():
        if axis == "Q":
            datacube.bin_Q(N, dtype=dtype)
        else:
            datacube.bin_R(N)
            datacube.data = datacube.data.astype(dtype, copy=False)


def hot_pixel_mask(mean_pattern, thresh):
    """
    The hot pixels: those in the mean pattern brighter by more than `thresh`
    than the second brightest pixel around them, as py4DSTEM's
    filter_hot_pixels finds them. That works on the mean pattern alone, so
    it's run on a 1 x 1 datacube of just that -- giving the mask without
    touching the data -- or the mean pattern, which it filters too, so it's
    given a copy.
    """
    probe = py4DSTEM.DataCube(np.array(mean_pattern, dtype=np.float64)[None, None])
    with _quiet():
        _, mask = filter_hot_pixels(probe, thresh, return_mask=True)
    return mask


def dead_pixel_mask(mean_pattern, fraction):
    """
    The dead pixels: those in the mean pattern dimmer than `fraction` of the
    median of the 8 pixels around them. (py4DSTEM has nothing to find them.)
    """
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    neighbours = median_filter(mean_pattern, footprint=footprint, mode="nearest")
    return mean_pattern < fraction * neighbours


def replace_pixels(datacube, mask):
    """
    Replace the masked pixels in every pattern, in place, with the median of
    the pixels around them. Returns their original values, for
    restore_pixels.
    """
    original = datacube.data[:, :, mask]
    with _quiet():
        datacube.median_filter_masked_pixels(mask, kernel_width=KERNEL_WIDTH)
    return original


def restore_pixels(datacube, mask, original):
    datacube.data[:, :, mask] = original


def masked_mean(data, mask):
    """The mean over the scan of each masked pixel, as an array of them."""
    return np.mean(data[:, :, mask], axis=(0, 1))
