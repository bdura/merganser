#!/usr/bin/env python
import rospy
import duckietown_utils as dtu

from duckietown_msgs.msg import Vector2D

from merganser_msgs.msg import SkeletonMsg, SkeletonsMsg
from merganser_ground_projection.utils.factories import get_ground_projection_geometry_for_robot


class GroundProjectionNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.loginfo('Initializing...')

        self.robot_name = self.get_robot_name('~veh_name')
        self.geometry = get_ground_projection_geometry_for_robot(self.robot_name)

        # Subscribers
        self.sub_skeleton = rospy.Subscriber("~skeleton_in",
                                             SkeletonsMsg,
                                             self.process_msg)

        # Publishers
        self.pub_skeleton = rospy.Publisher("~skeleton_out",
                                            SkeletonsMsg,
                                            queue_size=1)

        self.loginfo('Initialized')

    def process_msg(self, skeletons_msg):
        skeletons = []
        for skeleton in skeletons_msg.skeletons:
            points = self.geometry.vectors_to_ground(skeleton.cloud)

            skeleton_out_msg = SkeletonMsg()
            skeleton_out_msg.color = skeleton.color
            skeleton_out_msg.cloud = [Vector2D(x=point.x, y=point.y)
                                      for point in points]
            skeletons.append(skeleton_out_msg)

        skeletons_out_msg = SkeletonsMsg()
        skeletons_out_msg.skeletons = skeletons
        self.pub_skeleton.publish(skeletons_msg)

    def loginfo(self, message):
        rospy.loginfo('[{0}] {1}'.format(self.node_name, message))

    def get_robot_name(self, name):
        robot_name = rospy.get_param(name, None)
        if robot_name is None:
            robot_name = dtu.get_current_robot_name()
        rospy.set_param(name, robot_name)
        return robot_name

    def on_shutdown(self):
        self.loginfo('Shutting down')


if __name__ == '__main__':
    rospy.init_node('merganser_ground_projection_node', anonymous=False)
    ground_projection_node = GroundProjectionNode()
    rospy.on_shutdown(ground_projection_node.on_shutdown)
    rospy.spin()
