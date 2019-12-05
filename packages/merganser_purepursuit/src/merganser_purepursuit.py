#!/usr/bin/env python

import numpy as np
import rospy
from duckietown_msgs.msg import SegmentList, BoolStamped, Twist2DStamped
from duckietown_utils.instantiate_utils import instantiate
from geometry_msgs.msg import Point
from std_msgs.msg import Float32


def collapse(angle):
    angle %= 2 * np.pi
    angle += 2 * np.pi * (angle < - np.pi)
    angle -= 2 * np.pi * (angle > np.pi)
    return angle


class PurePursuitNode(object):

    def __init__(self):
        self.node_name = "PurePursuit Node"

        self.v0 = 1.5

        self.idle = True

        # Subscriber
        self.sub_waypoint = rospy.Subscriber('~waypoint', Point, self.process_waypoint)

        # Publisher
        self.pub_command = rospy.Publisher('~command', Twist2DStamped, queue_size=1)

        # timer for updating the params
        self.timer = rospy.Timer(rospy.Duration.from_sec(2.0), self.update_params)

        # self.publish_command(.1, 0)

        # We need to start the simulation...
        rospy.Timer(rospy.Duration.from_sec(2.0), self.kickstart)

    def kickstart(self, event):
        if self.idle:
            self.publish_command(.1, 0)

    def update_params(self, _event):

        self.v0 = rospy.get_param('~v0', 3)

    def process_waypoint(self, point):
        self.idle = False

        x, y = point.x, point.y

        alpha = np.arctan2(y, x)
        alpha = collapse(alpha)

        v = max(self.v0 * np.abs(np.cos(alpha)), .2)

        omega = 2 * v * np.sin(alpha) / np.sqrt(x ** 2 + y ** 2)

        self.publish_command(v, omega)

    def publish_command(self, v, omega):
        if v == 0.:
            return
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
