#!/usr/bin/env python
import json

import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown_msgs.msg import SegmentList, LanePose, BoolStamped, Twist2DStamped, FSMState
from duckietown_utils.instantiate_utils import instantiate
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, String


def collapse(angle):
    angle %= 2 * np.pi
    angle += 2 * np.pi * (angle < - np.pi)
    angle -= 2 * np.pi * (angle > np.pi)
    return angle


class PurePursuitNode(object):

    def __init__(self):
        self.node_name = "PurePursuit Node"

        self.v0 = .1

        # Subscriber
        self.sub_waypoint = rospy.Subscriber('~waypoint', Point, self.process_waypoint)

        # Publisher
        self.pub_command = rospy.Publisher('~command', Twist2DStamped, queue_size=1)

        # timer for updating the params
        self.timer = rospy.Timer(rospy.Duration.from_sec(2.0), self.update_params)

        self.prev_v, self.prev_omega = self.v0, 0.

    def update_params(self, _event):

        self.v0 = rospy.get_param('~v0', .3)

    def process_waypoint(self, point):
        self.publish_command(self.prev_v, self.prev_omega)

        x, y = point.x, point.y

        alpha = np.arctan2(y, x)

        v = self.v0 / (1 + alpha ** 2)
        omega = 2 * v * np.sin(alpha) / np.sqrt(x ** 2 + y ** 2)

        self.prev_v, self.prev_omega = v, omega

        self.publish_command(v, omega)

    def publish_command(self, v, omega):
        message = Twist2DStamped()
        message.v = v
        message.omega = omega
        self.pub_command.publish(message)

    def on_shutdown(self):
        self.publish_command(0, 0)
        rospy.log('Shutting down.')

    def log(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    rospy.init_node('merganser_purepursuit_node', anonymous=False)
    pure_pursuit = PurePursuitNode()
    rospy.on_shutdown(pure_pursuit.on_shutdown)
    rospy.spin()
