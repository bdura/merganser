#!/usr/bin/env python
import numpy as np
import cv2
import rospy
import duckietown_utils as dtu

from duckietown_msgs.msg import Twist2DStamped

from merganser_msgs.msg import SkeletonsMsg
from lf_pure_pursuit.utils import follow_point_to_velocities, Velocities


class PurePursuitNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.loginfo('Initializing...')

        # Publishers
        self.pub_car_cmd = rospy.Publisher('~car_cmd'
                                           Twist2DStamped,
                                           queue_size=1)

        # Subscribers
        self.sub_skeletons = rospy.Subscriber('~skeletons',
                                              SkeletonsMsg,
                                              self.update_cmd
                                              queue_size=1)

        # Maximal velocity
        v_bar_fallback = 0.25
        self.v_max = self.setup_parameter('~v_max', v_bar_fallback)

        # Look-ahead distance
        look_ahead_distance_fallback = 0.5
        self.look_ahead_distance = self.setup_parameter('~look_ahead_distance',
                                                        look_ahead_distance_fallback)

        # Offset
        offset_fallback = 0.3
        self.offset = self.setup_parameter('~offset', offset_fallback)

        # Minimal sinus alpha (related to the gain for the velocity)
        sin_alpha_min_fallback = 0.3
        self.sin_alpha_min = self.setup_parameter('~sin_alpha_min',
                                                  sin_alpha_min_fallback)

        # Gain for omega
        gain_omega_fallback = 2.
        self.gain_omega = self.setup_parameter('~gain_omega', gain_omega_fallback)

        # Gain for minimum velocity
        gain_min_velocity_fallback = 0.5
        self.gain_min_velocity = self.setup_parameter('~gain_min_velocity',
                                                      gain_min_velocity_fallback)

        # Smoothing
        decay_rate_fallback = 0.9
        self.decay_rate = self.setup_parameter('~decay_rate', decay_rate_fallback)

        self.previous = Velocities(v=self.v_max, omega=0.)
        self.loginfo('Initialized')

    def update_cmd(self, skeletons_msg):
        follow_point = None # TODO
        velocities = follow_point_to_velocities(follow_point,
                                                self.previous,
                                                v_max=self.v_max,
                                                decay_rate=self.decay_rate,
                                                gain_min_velocity=self.gain_min_velocity,
                                                gain_omega=self.gain_omega,
                                                sin_alpha_min=self.sin_alpha_min)
        self.previous = velocities
        self.publish_cmd(velocities)

    def publish_cmd(self, velocities):
        car_cmd_msg = Twist2DStamped()
        car_cmd_msg.v = velocities.v
        car_cmd_msg.omega = velocities.omega

        self.pub_car_cmd.publish(car_cmd_msg)

    def setup_parameter(self, name, default):
        value = rospy.get_param(name, default)
        rospy.set_param(name, value)
        return value

    def loginfo(self, message):
        rospy.loginfo('[{0}] {1}'.format(self.node_name, message))

    def on_shutdown(self):
        self.sub_skeletons.unregister()
        self.publish_cmd(0., 0.)

        rospy.sleep(0.5)
        self.loginfo('Shutting down')


if __name__ == '__main__':
    rospy.init_node('lf_pure_pursuit_node', anonymous=False)
    purepursuit_node = PurePursuitNode()
    rospy.on_shutdown(purepursuit_node.on_shutdown)
    rospy.spin()
