import numpy as np
import py4DSTEM
from py4DSTEM.process.calibration import get_probe_size

# Most scratch memory a virtual image computation converts to float at once.
CHUNK_BYTES = 128 * 2**20


def aperture_mask(Qshape, apertures):
    """
    The union of circular apertures, each given as ((qx, qy), radius) in
    diffraction-space pixels, as a boolean mask of shape Qshape.
    """
    mask = np.zeros(Qshape, dtype=np.bool_)
    for (qx, qy), radius in apertures:
        mask |= py4DSTEM.DataCube.make_detector(
            Qshape, "circle", ((qx, qy), float(radius))
        )
    return mask


def masked_sum(data, mask):
    """
    Virtual image: each diffraction pattern summed over a detector mask.

    A vectorised stand-in for DataCube.get_virtual_image, which loops over
    scan positions in Python (and attaches its result to the datacube's
    tree). Only the mask's bounding box is read, a block of scan rows at a
    time, so converting to float never needs more than CHUNK_BYTES -- and
    `data` can be anything that supports slicing (numpy, memmap, h5py).
    """
    R_Nx, R_Ny = data.shape[:2]
    vimg = np.zeros((R_Nx, R_Ny))

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return vimg
    slice_x = slice(rows[0], rows[-1] + 1)
    slice_y = slice(cols[0], cols[-1] + 1)
    weights = mask[slice_x, slice_y].astype(np.float32)

    step = max(1, CHUNK_BYTES // (R_Ny * weights.size * 4))
    for rx in range(0, R_Nx, step):
        block = np.asarray(data[rx : rx + step, :, slice_x, slice_y], dtype=np.float32)
        vimg[rx : rx + step] = np.tensordot(block, weights, axes=2)
    return vimg


def mean_dp(data):
    """
    The mean diffraction pattern, read a block of scan rows at a time (like
    masked_sum). Unlike DataCube.get_dp_mean, it doesn't attach anything to
    the datacube's tree.
    """
    R_Nx, R_Ny, Q_Nx, Q_Ny = data.shape
    total = np.zeros((Q_Nx, Q_Ny))
    step = max(1, CHUNK_BYTES // (R_Ny * Q_Nx * Q_Ny * 8))
    for rx in range(0, R_Nx, step):
        total += np.sum(data[rx : rx + step], axis=(0, 1), dtype=np.float64)
    return total / (R_Nx * R_Ny)


def measure_bf_radius(data):
    """
    The bright-field disk radius in pixels, measured from the mean
    diffraction pattern. Calls py4DSTEM's get_probe_size directly, as the
    DataCube method writes the result into the datacube's calibration.
    """
    radius, _, _ = get_probe_size(mean_dp(data))
    return float(radius)
