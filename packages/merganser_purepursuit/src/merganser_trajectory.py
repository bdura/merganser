#!/usr/bin/env python
from enum import Enum

import numpy as np
import rospy
from geometry_msgs.msg import Point
from merganser_msgs.msg import BeziersMsg
from merganser_bezier.bezier import Bezier, compute_curve


class Color(Enum):
    WHITE = 0
    YELLOW = 1
    RED = 2


class TrajectoryNode(object):

    def __init__(self):

        self.node_name = "Trajectory Node"

        self.lookahead = 0.5
        self.correction = 0.5
        self.alpha = 0.1

        self.waypoint = np.zeros(2)

        # Update the parameters
        self.update_params()
        rospy.Timer(rospy.Duration.from_sec(2.0), self.update_params)

        # Subscriber
        self.sub_filtered_segments = rospy.Subscriber('~beziers', BeziersMsg, self.process_beziers)

        # Publisher
        self.pub_waypoint = rospy.Publisher('~waypoint', Point, queue_size=1)

    def update_params(self, _event=None):

        self.lookahead = rospy.get_param('~lookahead', .5)
        self.correction = rospy.get_param('~correction', .5)
        self.alpha = rospy.get_param('~threshold', .2)

    def get_bezier_curve(self, message):
        controls, color = message.controls, message.color

        c = np.array([[p.x, p.y] for p in controls])

        # Flips the controls so the curve points outward
        if c[0, 0] > c[-1, 0]:
            c = c[[3, 2, 1, 0]]

        b = Bezier.from_controls(c)

        return b, color

    def process_beziers(self, message):
        points = []

        if len(message.beziers) == 0:
            return

        for m in message.beziers:
            b, color = self.get_bezier_curve(m)

            p = b()
            n = b.normal()

            if Color(color) == Color.WHITE:
                points.append(p + self.correction * n)
            elif Color(color) == Color.RED:
                points.append(p - self.correction * n)

        waypoints = np.vstack(points)

        weights = np.exp((np.linalg.norm(waypoints, axis=1, keepdims=True) - self.lookahead))
        weights /= weights.sum()

        waypoint = (waypoints * weights).sum(axis=0)

        self.waypoint = waypoint * self.alpha + self.waypoint * (1 - self.alpha)

        w = Point(self.waypoint[0], self.waypoint[1], 0)
        self.pub_waypoint.publish(w)

    def log(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))

    def on_shutdown(self):
        rospy.loginfo("[LaneFilterNode] Shutdown.")


if __name__ == '__main__':
    rospy.init_node('merganser_trajectory_node', anonymous=False)
    trajectory_node = TrajectoryNode()
    rospy.spin()
