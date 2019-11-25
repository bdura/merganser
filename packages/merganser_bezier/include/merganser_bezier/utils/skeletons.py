import numpy as np


def _extract_skeleton(skeleton):
    cloud = np.array([[point.x, point.y] for point in skeleton.cloud])
    color = skeleton.color

    return cloud, color
