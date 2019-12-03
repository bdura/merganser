#!/usr/bin/env python
import numpy as np
import rospy
import duckietown_utils as dtu

from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PointStamped, Point

from merganser_visualization.skeletons import skeletons_to_marker_array
from merganser_visualization.bezier import beziers_to_marker_array
from merganser_msgs.msg import SkeletonsMsg, BeziersMsg


class VisualizationNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.loginfo('Initializing...')

        self.veh_name = self.setup_parameter('~veh_name', 'default')

        # Publishers
        self.pub_skeletons = rospy.Publisher('~skeletons_markers',
                                             MarkerArray,
                                             queue_size=1)
        self.pub_beziers = rospy.Publisher('~beziers_markers',
                                           MarkerArray,
                                           queue_size=1)
        self.pub_waypoint = rospy.Publisher('~waypoint_marker',
                                            PointStamped,
                                            queue_size=1)

        # Subscribers
        self.sub_skeletons = rospy.Subscriber('~skeletons',
                                              SkeletonsMsg,
                                              self.callback_skeletons)
        self.sub_beziers = rospy.Subscriber('~beziers',
                                            BeziersMsg,
                                            self.callback_beziers)
        self.sub_waypoint = rospy.Subscriber('~waypoint',
                                             Point,
                                             self.callback_waypoint)

        self.loginfo('Initialized')

    def callback_skeletons(self, skeletons_msg):
        marker_array = skeletons_to_marker_array(skeletons_msg,
                                                 veh_name=self.veh_name)
        self.pub_skeletons.publish(marker_array)

    def callback_beziers(self, beziers_msg):
        marker_array = beziers_to_marker_array(beziers_msg,
                                               veh_name=self.veh_name)
        self.pub_beziers.publish(marker_array)

    def callback_waypoint(self, waypoint_msg):
        point_msg = PointStamped()
        point_msg.header.frame_id = self.veh_name
        point_msg.point = waypoint_msg
        self.pub_waypoint.publish(point_msg)

    def loginfo(self, message):
        rospy.loginfo('[{0}] {1}'.format(self.node_name, message))

    def setup_parameter(self, name, default):
        value = rospy.get_param(name, default)
        return value

    def on_shutdown(self):
        self.loginfo('Shutting down')


if __name__ == '__main__':
    rospy.init_node('merganser_visualization_node', anonymous=False)
    visualization_node = VisualizationNode()
    rospy.on_shutdown(visualization_node.on_shutdown)
    rospy.spin()
