import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import io

from .skeletons import extract_skeleton


def to_coordinate(coords):
    coords = coords.T[[1, 0]]
    coords = coords * np.array([[-1.], [1.]])
    return coords


def plt2img():
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=150)
    buf.seek(0)

    im = Image.open(buf)
    img = np.array(im)

    buf.close()
    plt.close()

    return img[..., [2, 1, 0]]


def plot_fitted_skeleton(beziers, skeletons):

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('black')

    ax.scatter([0], [0], color='white')

    for i, (b, s) in enumerate(zip(beziers, skeletons)):
        c_, _ = extract_skeleton(s)

        ax.scatter(*to_coordinate(c_), alpha=.5, color=b.color)  # 'C' + str(i+1))

        ax.plot(*to_coordinate(b()), alpha=1, lw=5, color=b.color)  # 'C' + str(i+1))
        ax.scatter(*to_coordinate(b.controls), alpha=1, color=b.color)  # 'C' + str(i+1))

    return plt2img()


def plot_waypoint(beziers, waypoint, waypoints):

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('black')

    ax.scatter([0], [0], color='white')

    for i, b in enumerate(beziers):
        ax.plot(*to_coordinate(b()), alpha=1, lw=5, color=b.color, zorder=1)  # 'C'+str(i))

    ax.plot(*to_coordinate(waypoints), alpha=1, lw=5, color='green', zorder=2)

    x, y = waypoint
    ax.scatter([-y], [x], color='red', zorder=3)

    plt.axis('scaled')

    return plt2img()


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
