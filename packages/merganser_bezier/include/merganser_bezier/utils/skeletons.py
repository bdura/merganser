import numpy as np
from enum import Enum

from merganser_msgs.msg import BezierMsg, SkeletonsMsg, BeziersMsg


class Color(Enum):
    white = 0
    yellow = 1
    red = 2


def extract_skeleton(skeleton):
    cloud = np.array([[point.x, point.y] for point in skeleton.cloud])
    color = Color(skeleton.color)

    return cloud, color


def make_bezier_message(bezier):
    if not bezier.fitted:
        return None
    msg = BezierMsg()
    for i, c in enumerate(bezier.controls):
        msg.controls[i].x = c[0]
        msg.controls[i].y = c[1]
    return msg
