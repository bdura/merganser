import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import io

from .skeletons import _extract_skeleton


def plot_fitted_skeleton(beziers, skeletons):
    def to_coordinate(coords):
        coords = coords.T[[1, 0]]
        coords = coords * np.array([[1.], [-1.]])
        return coords

    plt.figure()

    for i, (b, s) in enumerate(zip(beziers, skeletons)):
        c_, _ = _extract_skeleton(s)

        c = b()
        n = b.normal()

        plt.scatter([0], [0], color='black')

        plt.scatter(
            *to_coordinate(np.vstack((c + .2 * n, c - .2 * n))),
            alpha=.5,
            color='C0',
            s=10 * np.exp((np.linalg.norm(c + 1 * n, axis=1) - .5) ** 2)
        )

        plt.scatter(*to_coordinate(c_), alpha=.5, color='C' + str(i+1))

        plt.plot(*to_coordinate(b()), alpha=1, color='C' + str(i+1))
        plt.scatter(*to_coordinate(b.controls), alpha=1, color='C' + str(i+1))

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=150)
    buf.seek(0)

    im = Image.open(buf)
    img = np.array(im)

    buf.close()
    plt.close()

    return img


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tostring())
