#!/usr/bin/env python
# import cv2
import duckietown_utils as dtu
import rospy
import numpy as np
import time
from merganser_bezier.bezier import Bezier, compute_curve
from merganser_bezier.utils.plots import plot_fitted_skeleton
from merganser_msgs.msg import BezierMsg, SkeletonMsg, SkeletonsMsg, BeziersMsg
from duckietown_msgs.msg import Vector2D, Twist2DStamped

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class BezierNode(object):
    def __init__(self):
        self.node_name = 'BezierNode'

        self.stats = Stats()

        self.intermittent_interval = 100
        self.intermittent_counter = 0

        # Parameters
        self.verbose = False
        self.refit_every = 1
        self.loss_threshold = .01
        self.reg = 5e-3
        self.lr = .1
        self.extension = .1
        self.curve_precision = 20
        self.fitting_steps = 20
        self.eps = 1e-3

        # Subscribers
        self.sub_skeleton = rospy.Subscriber(
            '~skeletons',
            SkeletonsMsg,
            self.process_skeletons,
            queue_size=1
        )

        self.sub_skeleton = rospy.Subscriber(
            '~command',
            Twist2DStamped,
            self.update_commands,
            queue_size=1
        )

        self.dx = 0
        self.dtheta = 0

        self.time = time.time()

        # Publishers
        self.pub_bezier = rospy.Publisher('~beziers', BeziersMsg, queue_size=1)

        self.update_params(None)

        # Updates the parameters every 2 seconds
        rospy.Timer(rospy.Duration.from_sec(2.), self.update_params)

        # Attributes
        self.steps = 0
        self.beziers = []

        self.time = 0
        self.n = 0

        self.bridge = CvBridge()
        self.pub_skeletons_image = rospy.Publisher('~curves', Image, queue_size=1)

    def update_params(self, _event):
        self.verbose = rospy.get_param('~verbose', False)

        self.refit_every = rospy.get_param('~refit_every', 1)
        self.intermittent_interval = rospy.get_param('~intermittent_interval', 100)

        self.loss_threshold = rospy.get_param('~loss_threshold', 1e-9)

        self.extension = rospy.get_param('~extension', 0)

        self.curve_precision = rospy.get_param('~curve_precision', 20)
        self.fitting_steps = rospy.get_param('~fitting_steps', 200)

        self.eps = rospy.get_param('~eps', 1e-3)
        self.lr = rospy.get_param('~lr', 1e-2)
        self.reg = rospy.get_param('~reg', 1e-2)

    @property
    def dt(self):
        t = time.time()
        dt = t - self.time
        self.time = t
        return dt

    def update_commands(self, msg):
        v, omega = msg.v, msg.omega
        self.dx = v * self.dt
        self.dtheta = omega * self.dt

    def loginfo(self, message):
        rospy.loginfo('[%s] %s' % (self.node_name, message))

    def _extend_beziers(self):
        for bezier in self.beziers:
            bezier.extrapolate(0 - self.extension, 1 + self.extension)

    def _extract_skeleton(self, skeleton):
        cloud = np.array([[point.x, point.y] for point in skeleton.cloud])
        color = skeleton.color

        return cloud, color

    def _process_skeleton(self, skeleton):

        cloud, color = self._extract_skeleton(skeleton)

        for bezier in self.beziers:
            bezier.predict(self.dx, self.dtheta)

        losses = np.array([b.loss(cloud) for b in self.beziers if b.color == color])
        argmin = losses.argmin() if len(losses) > 0 else -1

        if argmin > -1 and losses[argmin] < self.loss_threshold:
            bezier = self.beziers[argmin].copy()
            bezier.correct(cloud)
        else:
            bezier = Bezier(4, self.curve_precision, reg=self.reg, color=color)
            bezier.collapse(cloud)

        return bezier, color

    def _make_bezier_message(self, bezier, color):
        msg = BezierMsg()
        msg.color = color
        for i, c in enumerate(bezier.controls):
            msg.controls[i].x = c[0]
            msg.controls[i].y = c[1]
        return msg

    def process_skeletons(self, skeletons_msg):

        t0 = time.time()

        self.stats.received()

        # Extends all collected Bezier curves
        self._extend_beziers()

        # Gets the skeletons
        skeletons = skeletons_msg.skeletons

        # Creates the message containing the updated Bezier curves
        beziers = []
        messages = BeziersMsg()

        # Removes scattered skeletons
        skeletons = [skeleton for skeleton in skeletons if len(skeleton.cloud) > 10]

        for skeleton in skeletons:

            bezier, color = self._process_skeleton(skeleton)
            beziers.append(bezier)

            # Creates the message associated with the bezier curve and appends it to the general message
            msg = self._make_bezier_message(bezier, color)
            messages.beziers.append(msg)

        self.beziers = beziers

        if self.verbose and self.intermittent_log_now():
            img = plot_fitted_skeleton(beziers, skeletons)
            img_message = self.bridge.cv2_to_imgmsg(img, 'bgr8')

            self.pub_skeletons_image.publish(img_message)

        self.stats.processed()
        self.intermittent_counter += 1
        if self.intermittent_log_now():
            self.intermittent_log(self.stats.info())
            self.stats.reset()

        self.pub_bezier.publish(messages)

        t = time.time() - t0

        self.time = self.time * .95 + t * .05

    def intermittent_log_now(self):
        return self.intermittent_counter % self.intermittent_interval == 1

    def intermittent_log(self, s):
        if not self.intermittent_log_now():
            return
        self.loginfo('%3d:%s' % (self.intermittent_counter, s))
        self.loginfo('Mean process time : %.2f ms.' % (self.time * 1000))

        self.loginfo('verbose %s' % (self.verbose,))
        self.loginfo('refit_every %s' % (self.refit_every,))
        self.loginfo('loss_threshold %s' % (self.loss_threshold,))
        self.loginfo('reg %s' % (self.reg,))
        self.loginfo('lr %s' % (self.lr,))
        self.loginfo('extension %s' % (self.extension,))
        self.loginfo('curve_precision %s' % (self.curve_precision,))
        self.loginfo('fitting_steps %s' % (self.fitting_steps,))
        self.loginfo('eps %s' % (self.eps,))

    def on_shutdown(self):
        self.loginfo('Shutdown...')


class Stats():
    def __init__(self):
        self.nresets = 0
        self.reset()

    def reset(self):
        self.nresets += 1
        self.t0 = time.time()
        self.nreceived = 0
        self.nskipped = 0
        self.nprocessed = 0

    def received(self):
        if self.nreceived == 0 and self.nresets == 1:
            rospy.loginfo('line_detector_node received first image.')
        self.nreceived += 1

    def skipped(self):
        self.nskipped += 1

    def processed(self):
        if self.nprocessed == 0 and self.nresets == 1:
            rospy.loginfo('line_detector_node processing first image.')

        self.nprocessed += 1

    def info(self):
        delta = time.time() - self.t0

        if self.nreceived:
            skipped_perc = (100.0 * self.nskipped / self.nreceived)
        else:
            skipped_perc = 0

        def fps(x):
            return '%.1f fps' % (x / delta)

        m = ('In the last %.1f s: received %d (%s) processed %d (%s) skipped %d (%s) (%1.f%%)' %
             (delta, self.nreceived, fps(self.nreceived),
              self.nprocessed, fps(self.nprocessed),
              self.nskipped, fps(self.nskipped), skipped_perc))
        return m


if __name__ == '__main__':
    rospy.init_node('merganser_bezier_node', anonymous=False)
    bezier_node = BezierNode()
    rospy.on_shutdown(bezier_node.on_shutdown)
    rospy.spin()
