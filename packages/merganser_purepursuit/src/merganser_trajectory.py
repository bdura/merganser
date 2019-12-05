#!/usr/bin/env python
from enum import Enum

import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from merganser_msgs.msg import BeziersMsg
from merganser_bezier.bezier import Bezier, compute_curve
from duckietown_msgs.msg import Vector2D, Twist2DStamped

from merganser_bezier.utils.plots import plot_waypoint
from merganser_visualization.general import line_to_marker, color_to_rgba

import time


class Color(Enum):
    WHITE = 0
    YELLOW = 1
    RED = 2


def rotation(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    r = np.array([[cos, -sin],
                  [sin, cos]])
    return r


def get_bezier_curve(message):
    if message.fitted == 0:
        return None
    controls = message.controls
    c = np.array([[p.x, p.y] for p in controls])
    b = Bezier.from_controls(c)
    return b


class TrajectoryNode(object):

    def __init__(self):

        self.node_name = "Trajectory Node"
        self.veh_name = rospy.get_param('~veh', 'default')

        self.lookahead = 0.2
        self.correction = 0.2
        self.alpha = 0.1

        self.verbose = False

        self.iters = 0

        self.waypoint = np.zeros(2)

        self.t = time.time()

        self.dx = 0
        self.dtheta = 0

        # Update the parameters
        self.update_params()
        rospy.Timer(rospy.Duration.from_sec(2.0), self.update_params)

        self.bridge = CvBridge()

        # Publisher
        self.pub_waypoint = rospy.Publisher('~waypoint', Point, queue_size=1)
        self.pub_image = rospy.Publisher('~image', Image, queue_size=1)
        self.pub_trajectory_marker = rospy.Publisher('~trajectory_marker', Marker, queue_size=1)

        # Subscriber
        self.sub_filtered_segments = rospy.Subscriber('~beziers', BeziersMsg, self.process_beziers)

        self.sub_commands = rospy.Subscriber(
            '~command',
            Twist2DStamped,
            self.update_commands,
            queue_size=1
        )

    def update_params(self, _event=None):

        self.lookahead = rospy.get_param('~lookahead', .4)
        self.correction = rospy.get_param('~correction', .25)
        self.alpha = rospy.get_param('~alpha', .8)
        self.veh_name = rospy.get_param('~veh', 'default')

        self.verbose = rospy.get_param('~verbose', False)

    def update_commands(self, msg):
        v, omega = msg.v, msg.omega
        self.dx = v * self.dt
        self.dtheta = omega * self.dt

    @property
    def dt(self):
        t = time.time()
        dt = t - self.t
        self.t = t
        return dt

    def update_waypoint(self):
        # Constructs the rotation matrix
        r = rotation(- self.dtheta)

        # Constructs the offset
        offset = np.zeros(2)
        offset[0] = - self.dx

        self.waypoint = np.matmul(r, self.waypoint + offset)

    def process_beziers(self, message):

        # Update waypoint using kinematics
        self.update_waypoint()

        left = get_bezier_curve(message.left)
        yellow = get_bezier_curve(message.yellow)
        right = get_bezier_curve(message.right)

        if left is None and yellow is None and right is None:
            return

        if yellow is not None and right is not None:
            waypoints = (yellow() + right()) / 2

        elif yellow is not None:
            waypoints = yellow() - self.correction * yellow.normal() * .9

        elif right is not None:
            waypoints = right() + self.correction * right.normal() * .9

        else:
            waypoints = left() - 2 * self.correction * left.normal() * .9

        arg = np.abs(np.linalg.norm(waypoints, axis=1) - self.lookahead).argmin()

        waypoint = waypoints[arg]

        self.waypoint += self.alpha * (waypoint - self.waypoint)

        w = Point(self.waypoint[0], self.waypoint[1], 0)
        self.pub_waypoint.publish(w)

        marker = line_to_marker(waypoints,
                                color_to_rgba('green'),
                                name='trajectory_curve',
                                veh_name=self.veh_name)
        self.pub_trajectory_marker.publish(marker)

        self.iters += 1

        if self.verbose and self.iters % 10 == 0:
            img = plot_waypoint(beziers, self.waypoint, waypoints)
            img_message = self.bridge.cv2_to_imgmsg(img, 'bgr8')

            self.pub_image.publish(img_message)

    def log(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))

    def on_shutdown(self):
        self.log("Shutting down.")


if __name__ == '__main__':
    rospy.init_node('merganser_trajectory_node', anonymous=False)
    trajectory_node = TrajectoryNode()
    rospy.spin()
