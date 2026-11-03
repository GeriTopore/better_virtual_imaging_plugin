"""
Figures for saving a preview: its diffraction pattern, with the apertures
drawn on it, beside the virtual image they make -- each as the preview
shows it, with the same scaling, levels and colormap.
"""

import io

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch

from .lattice import APERTURE_FILL_ALPHA

# Dots per inch for saving raster formats.
SAVE_DPI = 200

# Inches wide for each image, and for the merged view's colour key.
IMAGE_WIDTH = 5.0
KEY_WIDTH = 1.8


def _mpl_cmap(pg_cmap):
    # A pyqtgraph ColorMap, as a matplotlib one.
    return ListedColormap(pg_cmap.getLookupTable(0.0, 1.0, 256, alpha=False) / 255)


def _view_image(view):
    """
    What an ImageView shows: its image (displayed transposed, as the main
    window's are, so transposed back -- keeping any colour axis last), its
    levels, and its colormap.
    """
    image = view.image
    image = np.transpose(image, (1, 0, 2)) if image.ndim == 3 else image.T
    levels = view.getImageItem().getLevels()
    cmap = view.ui.histogram.gradient.colorMap()
    return image, levels, cmap


def _show(ax, view, title):
    image, levels, cmap = _view_image(view)
    if image.ndim == 3:  # RGB, already scaled
        ax.imshow(image, interpolation="nearest")
    else:
        vmin, vmax = (None, None) if levels is None else levels
        ax.imshow(
            image,
            cmap=_mpl_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def pane_figure(pane, title, dp_title, vi_title):
    """
    A figure of a pane: the diffraction pattern with its apertures (see
    PreviewPane.aperture_drawings) and the virtual image -- or, for the
    merged view, the colour composite, with a key to its colours.
    """
    # The key, if there is one, gets a column of its own to the right, so
    # the two images stay the same size.
    legend = getattr(pane, "legend_entries", None)
    widths = [IMAGE_WIDTH, IMAGE_WIDTH] + ([KEY_WIDTH] if legend else [])
    fig = Figure(figsize=(sum(widths), 5.4), layout="constrained")
    axes = fig.subplots(1, len(widths), gridspec_kw={"width_ratios": widths})
    ax_dp, ax_vi = axes[:2]
    fig.suptitle(title, fontsize=12)

    _show(ax_dp, pane.dp_view, dp_title)
    # The pattern is shown as an image with qx down and qy across, so data
    # pixel (qx, qy) is at (qy, qx).
    for (qx, qy), radius, rgb in pane.aperture_drawings():
        ax_dp.add_patch(
            Circle(
                (qy, qx),
                radius,
                facecolor=(*rgb, APERTURE_FILL_ALPHA),
                edgecolor=rgb,
                linewidth=1.5,
            )
        )

    _show(ax_vi, pane.vi_view, vi_title)
    if legend:
        # To the right of the composite, one entry per line.
        ax_key = axes[2]
        ax_key.set_axis_off()
        ax_key.legend(
            handles=[Patch(color=rgb, label=name) for name, rgb in legend],
            loc="center left",
            ncol=1,
            frameon=False,
            fontsize=10,
        )
    return fig


def render_png(fig, dpi):
    """The figure as PNG bytes, e.g. for previewing it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


def supported_formats(fig):
    """The file extensions the figure can be saved as, e.g. "png"."""
    return set(fig.canvas.get_supported_filetypes())
