import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3

from merganser_visualization.general import color_to_rgba


def skeleton_to_marker(skeleton, index, veh_name='default'):
    marker = Marker()
    marker.header.frame_id = veh_name

    marker.ns = '{0}/skeleton_points'.format(veh_name)
    marker.id = index
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration.from_sec(5.0)
    marker.type = Marker.POINTS
    marker.color = color_to_rgba(skeleton.color)
    marker.scale = Vector3(x=0.02, y=0.02, z=1.)

    for vector in skeleton.cloud:
        marker.points.append(Point(x=vector.x, y=vector.y, z=0.))

    return marker

def skeletons_to_marker_array(skeletons, veh_name='default'):
    markers = []
    for index, skeleton in enumerate(skeletons.skeletons):
        marker = skeleton_to_marker(skeleton, index, veh_name=veh_name)
        markers.append(marker)

    marker_array = MarkerArray()
    marker_array.markers = markers

    return marker_array
