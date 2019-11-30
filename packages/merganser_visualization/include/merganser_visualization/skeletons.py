import rospy
from visualization_msgs.msg import Marker, MarkerArray

from merganser_visualization.general import color_to_rgba


def skeleton_to_marker(skeleton, veh_name='default'):
    marker = Marker()
    marker.header.frame_id = veh_name

    marker.ns = '{0}/skeleton_points'.format(veh_name)
    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration.from_sec(5.0)
    marker.type = Marker.POINTS
    marker.color = color_to_rgba(skeleton.color)
    marker.scale.x = 0.02

    for vector in skeleton.cloud:
        # TODO: Convert vector to Vector3D?
        marker.points.append(vector)

    return marker

def skeletons_to_marker_array(skeletons, veh_name='default'):
    marker_array = MarkerArray()
    for skeleton in skeletons:
        marker = skeleton_to_marker(skeleton, veh_name=veh_name)
        marker_array.markers.append(marker)

    return marker_array
