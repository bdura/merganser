#!/usr/bin/env python
import numpy as np
import rospy
import duckietown_utils as dtu

from visualization_msgs.msg import MarkerArray

from merganser_visualization.skeletons import skeletons_to_marker_array
from merganser_msgs.msg import SkeletonsMsg


class VisualizationNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.loginfo('Initializing...')

        self.veh_name = self.setup_parameter('~veh_name', 'default')

        # Publishers
        self.pub_skeletons = rospy.Publishers('~skeletons_markers',
                                              MarkerArray,
                                              queue_size=1)

        # Subscribers
        self.sub_skeletons = rospy.Subscribers('~skeletons',
                                               SkeletonsMsg,
                                               self.callback_skeletons)

        self.loginfo('Initialized')

    def callback_skeletons(self, skeletons_msg):
        marker_array = skeletons_to_marker_array(skeletons_msg,
                                                 veh_name=self.veh_name)
        self.pub_skeletons.publish(marker_array)

    def loginfo(self, message):
        rospy.loginfo('[{0}] {1}'.format(self.node_name, message))

    def setup_parameter(self, name, default):
        value = rospy.get_param(name, default)
        rospy.set_param(name, value)
        return value

    def on_shutdown(self):
        self.loginfo('Shutting down')


if __name__ == '__main__':
    rospy.init_node('merganser_visualization_node', anonymous=False)
    visualization_node = VisualizationNode()
    rospy.on_shutdown(visualization_node.on_shutdown)
    rospy.spin()
