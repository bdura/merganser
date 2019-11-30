from std_msgs.msg import ColorRGBA
from merganser_msgs.msg import SkeletonMsg


def color_to_rgba(color):
    colors_to_rgba = {
        SkeletonMsg.WHITE: ColorRGBA(r=1., g=1., b=1., a=1.),
        SkeletonMsg.YELLOW: ColorRGBA(r=1., g=1., b=0., a=1.),
        SkeletonMsg.RED: ColorRGBA(r=1., g=0., b=0., a=1.)
    }

    if color not in colors_to_rgba:
        raise ValueError('Unknown color {0}'.format(color))
    return colors_to_rgba[color]
