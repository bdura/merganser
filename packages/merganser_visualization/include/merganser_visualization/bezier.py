import rospy
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3

from merganser_visualization.general import color_to_rgba
from merganser_bezier.utils.bernstein import compute_curve


def bezier_to_marker(bezier, index, veh_name='default'):
    marker = Marker()
    marker.header.frame_id = veh_name

    marker.ns = '{0}/beizer_curves'.format(veh_name)
    marker.id = index
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration.from_sec(5.0)
    marker.type = Marker.LINE_STRIP
    marker.color = color_to_rgba(bezier.color)
    marker.scale = Vector3(x=0.02, y=1., z=1.)

    controls = np.asarray([[vector.x, vector.y] for vector in bezier.controls])
    curve = compute_curve(controls, n=20)
    marker.points = [Point(x=point[0], y=point[1]) for point in curve]

    return marker

def beziers_to_marker_array(beziers, veh_name='default'):
    markers = []
    for index, bezier in enumerate(beziers.beziers):
        marker = bezier_to_marker(bezier, index, veh_name=veh_name)
        markers.append(marker)

    marker_array = MarkerArray()
    marker_array.markers = markers

    return marker_array
