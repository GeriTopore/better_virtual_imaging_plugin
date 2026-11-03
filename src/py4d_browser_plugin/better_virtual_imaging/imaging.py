import numpy as np
import py4DSTEM
from matplotlib.colors import hsv_to_rgb
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


def max_dp(data):
    """The maximum diffraction pattern, read a block of scan rows at a time."""
    R_Nx, R_Ny, Q_Nx, Q_Ny = data.shape
    result = None
    step = max(1, CHUNK_BYTES // (R_Ny * Q_Nx * Q_Ny * data.dtype.itemsize))
    for rx in range(0, R_Nx, step):
        block = np.max(data[rx : rx + step], axis=(0, 1))
        result = block if result is None else np.maximum(result, block)
    return result


def measure_bf_disk(mean_pattern):
    """
    The bright-field disk's radius and centre, (radius, qx0, qy0) in pixels,
    measured from the mean diffraction pattern. Calls py4DSTEM's
    get_probe_size directly, as the DataCube method writes the result into
    the datacube's calibration.
    """
    radius, qx0, qy0 = get_probe_size(mean_pattern)
    return float(radius), float(qx0), float(qy0)


def composite_colors(n):
    """
    Default RGB (0-1) colours for n images in a colour composite: hues evenly
    spaced around the colour wheel from red, at full saturation and value
    (as py4DSTEM.show(..., combine_images=True) uses).
    """
    return [tuple(float(c) for c in hsv_to_rgb((i / n, 1.0, 1.0))) for i in range(n)]


def _scale_ordered(image, vmin=0.02, vmax=0.98):
    # py4DSTEM.show's default intensity_range="ordered": scale to [0, 1]
    # between the values ranked at the vmin and vmax fractions, clipping.
    vals = np.sort(image[~np.isnan(image)])
    lo = vals[max(0, int(np.round((vals.size - 1) * vmin)))]
    hi = vals[min(vals.size - 1, int(np.round((vals.size - 1) * vmax)))]
    if hi == lo:
        lo, hi = vals[0], vals[-1]
    if hi == lo:
        return np.zeros(image.shape)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def color_composite(images, colors):
    """
    An RGB (0-1) colour composite of same-shaped images, each in its own RGB
    (0-1) colour: every image is scaled to [0, 1] as py4DSTEM.show does by
    default, tinted with its colour, and the tinted images are added,
    clipping at white where they overlap brightly.
    """
    rgb = np.zeros(images[0].shape + (3,))
    for image, color in zip(images, colors):
        rgb += _scale_ordered(image)[..., None] * np.asarray(color)
    return np.clip(rgb, 0.0, 1.0)
