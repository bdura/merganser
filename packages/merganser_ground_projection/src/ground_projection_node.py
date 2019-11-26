#!/usr/bin/env python

import duckietown_utils as dtu
import rospy
from merganser_msgs.msg import SkeletonsMsg
from merganser_ground_projection.ground_projection_interface import GroundProjection, \
    get_ground_projection_geometry_for_robot


class GroundProjectionNode(object):

    def __init__(self):
        self.node_name = "Merganser Ground Projection"
        self.active = True
        # self.bridge = CvBridge()

        robot_name = rospy.get_param("~config_file_name", None)

        if robot_name is None:
            robot_name = dtu.get_current_robot_name()

        self.robot_name = robot_name

        self.gpg = get_ground_projection_geometry_for_robot(self.robot_name)

        # Subs and Pubs
        self.sub_skeleton = rospy.Subscriber("~skeleton_in", SkeletonsMsg, self.process_msg)
        self.pub_skeleton = rospy.Publisher("~skeleton_out", SkeletonsMsg, queue_size=1)

    def process_msg(self, skeletons_msg):

        for skeleton in skeletons_msg.skeletons:
            for p in skeleton.cloud:

                # p.x, p.y = 1 - p.x, p.y
                # point = self.gpg.vector2ground(p)
                # print(point.x, point.x)

                # p.x, p.y = point.x, point.y
                pass

        self.pub_skeleton.publish(skeletons_msg)

    def onShutdown(self):
        rospy.loginfo("[MerganserGroundProjectionNode] Shutdown.")


if __name__ == '__main__':
    rospy.init_node('merganser_ground_projection', anonymous=False)
    ground_projection_node = GroundProjectionNode()
    rospy.on_shutdown(ground_projection_node.onShutdown)
    rospy.spin()
