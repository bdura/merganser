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
from merganser_bezier.utils.plots import plot_waypoint
from merganser_visualization.general import line_to_marker, color_to_rgba


class Color(Enum):
    WHITE = 0
    YELLOW = 1
    RED = 2


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

        # Update the parameters
        self.update_params()
        rospy.Timer(rospy.Duration.from_sec(2.0), self.update_params)

        # Subscriber
        self.sub_filtered_segments = rospy.Subscriber('~beziers', BeziersMsg, self.process_beziers)

        self.bridge = CvBridge()

        # Publisher
        self.pub_waypoint = rospy.Publisher('~waypoint', Point, queue_size=1)
        self.pub_image = rospy.Publisher('~image', Image, queue_size=1)
        self.pub_trajectory_marker = rospy.Publisher('~trajectory_marker', Marker, queue_size=1)

    def update_params(self, _event=None):

        self.lookahead = rospy.get_param('~lookahead', .3)
        self.correction = rospy.get_param('~correction', .1)
        self.alpha = rospy.get_param('~alpha', .8)
        self.veh_name = rospy.get_param('~veh', 'default')

        self.verbose = rospy.get_param('~verbose', False)

    def get_bezier_curve(self, message):
        controls, color = message.controls, message.color

        c = np.array([[p.x, p.y] for p in controls])

        # Flips the controls so the curve points outward
        if c[0, 0] > c[-1, 0]:
            c = c[[3, 2, 1, 0]]

        b = Bezier.from_controls(c, color=color)

        return b, color

    def process_beziers(self, message):
        points = []

        if len(message.beziers) == 0:
            return

        beziers = []

        for m in message.beziers:
            b, color = self.get_bezier_curve(m)

            p = b()
            n = b.normal()

            if Color(color) == Color.WHITE:
                points.append(p + self.correction * n)
            elif Color(color) == Color.RED:
                points.append(p - self.correction * n)

            beziers.append(b)

        yellows = [b for b in beziers if b.color == 'yellow']
        whites = [b for b in beziers if b.color == 'white']

        if len(yellows) > 0:
            yellow = yellows[0]
        else:
            yellow = None

        if len(whites) > 1:
            whites = sorted(whites, key=lambda b: b.controls[:, 1].mean())
            right, left = whites[0], whites[-1]
        elif len(whites) == 1:
            # If there is only one white line, we assume the one visible is on the right
            # The hypothesis is that we need to cross the yellow line for this to be the case
            right, left = whites[0], None
        else:
            right, left = None, None

        if yellow is not None \
                and right is not None \
                and yellow.controls[:, 1].mean() > right.controls[:, 1].mean():
            # waypoints = Bezier.from_controls((yellow.controls + right.controls) / 2, color='green')()
            waypoints = (yellow() + right()) / 2
        elif yellow is not None:
            waypoints = yellow() - self.correction * yellow.normal()
        elif right is not None:
            waypoints = right() + self.correction * right.normal()
        else:
            waypoints = None

        if waypoints is not None:

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
