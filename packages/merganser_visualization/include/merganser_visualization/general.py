import rospy

from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3

from merganser_msgs.msg import SkeletonMsg


def color_to_rgba(color):
    colors_to_rgba = {
        SkeletonMsg.WHITE: ColorRGBA(r=1., g=1., b=1., a=1.),
        SkeletonMsg.YELLOW: ColorRGBA(r=1., g=1., b=0., a=1.),
        SkeletonMsg.RED: ColorRGBA(r=1., g=0., b=0., a=1.),
        # Custom colors
        "green": ColorRGBA(r=0., g=1., b=0., a=1.)
    }

    if color not in colors_to_rgba:
        raise ValueError('Unknown color {0}'.format(color))
    return colors_to_rgba[color]

def line_to_marker(points, color, name='line', veh_name='default'):
    marker = Marker()
    marker.header.frame_id = veh_name

    marker.ns = '{0}/{1}'.format(veh_name, name)
    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration.from_sec(5.0)
    marker.type = Marker.LINE_STRIP
    marker.color = color
    marker.scale = Vector3(x=0.02, y=1., z=1.)

    for x, y in points:
        marker.points.append(Point(x=x, y=y, z=0.))

    return marker
