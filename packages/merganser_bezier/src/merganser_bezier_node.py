#!/usr/bin/env python
# import cv2
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown_msgs.msg import Twist2DStamped
from merganser_bezier.bezier import Bezier
from merganser_bezier.utils.plots import plot_fitted_skeleton
from merganser_bezier.utils.skeletons import extract_skeleton, Color, make_bezier_message
from merganser_msgs.msg import SkeletonsMsg, BeziersMsg
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

        self.dx = 0
        self.dtheta = 0

        self.t = time.time()

        self.update_params(None)

        # Updates the parameters every 2 seconds
        rospy.Timer(rospy.Duration.from_sec(2.), self.update_params)

        # Attributes
        self.steps = 0

        self.left = Bezier(4, 20)
        self.yellow = Bezier(4, 20)
        self.right = Bezier(4, 20)

        self.beziers = [self.left, self.yellow, self.right]

        self.time = 0
        self.n = 0

        self.bridge = CvBridge()

        # Publishers
        self.pub_bezier = rospy.Publisher('~beziers', BeziersMsg, queue_size=1)
        self.pub_skeletons_image = rospy.Publisher('~curves', Image, queue_size=1)

        # Subscribers
        self.sub_skeleton = rospy.Subscriber(
            '~skeletons',
            SkeletonsMsg,
            self.process,
            queue_size=1
        )

        self.sub_commands = rospy.Subscriber(
            '~command',
            Twist2DStamped,
            self.update_commands,
            queue_size=1
        )

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
        dt = t - self.t
        self.t = t
        return dt

    def update_commands(self, msg):
        v, omega = msg.v, msg.omega
        self.dx = v * self.dt
        self.dtheta = omega * self.dt

    def loginfo(self, message):
        rospy.loginfo('[%s] %s' % (self.node_name, message))

    def _extend_beziers(self):
        self.left.extrapolate(0 - self.extension, 1 + self.extension)
        self.right.extrapolate(0 - self.extension, 1 + self.extension)
        self.yellow.extrapolate(0 - self.extension, 1 + self.extension)

    def _predict(self):
        self.left.predict(self.dx, self.dtheta)
        self.yellow.predict(self.dx, self.dtheta)
        self.right.predict(self.dx, self.dtheta)

    def _process_yellows(self, yellows):
        if len(yellows) == 0:
            self.yellow.unfit()
        else:
            cloud = np.vstack(yellows)
            self.yellow.correct(cloud)

    def _process_whites(self, whites):

        if len(whites) == 0:
            self.left.unfit()
            self.right.unfit()
            return

        # Computes the centroids
        centroids = np.array([w.mean(axis=0) for w in whites])

        if self.yellow.fitted:
            # If the yellow line is already fitted, it becomes simple to differentiate left from right

            center = self.yellow.controls.mean(axis=0)
            normal = self.yellow.normal().mean(axis=0)

            dot_product = np.dot(centroids - center, normal)

            lefts = [w for w, d in zip(whites, dot_product) if d > 0.]
            rights = [w for w, d in zip(whites, dot_product) if d < -0.]

            lefts = sorted(lefts, key=lambda c: len(c), reverse=True)
            rights = sorted(rights, key=lambda c: len(c), reverse=True)

            left = lefts[0] if len(lefts) > 0 else None
            right = rights[0] if len(rights) > 0 else None

            if left is not None:
                self.left.correct(left)
            else:
                self.left.unfit()

            if right is not None:
                self.right.correct(right)
            else:
                self.right.unfit()

            return

        loss = np.array([
            [b.loss(cloud) for b in [self.left, self.right]]
            for cloud in whites
        ])

        if self.left.fitted and self.right.fitted:

            if len(whites) == 1:

                arg = loss[0].argmin()
                white = whites[0]
                if arg == 0:
                    self.left.correct(white)
                    self.right.unfit()
                else:
                    self.right.correct(white)
                    self.left.unfit()

            else:

                if len(loss) > 2:
                    argsort = loss.min(axis=1).argsort()
                    loss = loss[argsort[:2]]
                    whites = whites[argsort[:2]]

                lr_loss = loss.diagonal().sum()
                rl_loss = loss.sum() - lr_loss

                if lr_loss < rl_loss:
                    left, right = whites
                else:
                    right, left = whites

                self.left.correct(left)
                self.right.correct(right)

        elif self.left.fitted:
            loss = loss[:, 0]

            left = whites.pop(loss.argmin())
            self.left.correct(left)
            self.right.unfit()

        elif self.right.fitted:
            loss = loss[:, 1]

            right = whites.pop(loss.argmin())
            self.right.correct(right)
            self.left.unfit()

        else:
            # Otherwise we're screwed. Let's continue and hope for the best
            # ie that we'll see a yellow line at some point
            self.left.unfit()
            self.right.unfit()

    def process(self, skeletons_msg):

        t0 = time.time()

        self.stats.received()

        # Predicts and extends the bezier curves
        self._predict()
        # self._extend_beziers()

        # Gets the skeletons
        skeletons = skeletons_msg.skeletons
        skeletons = [extract_skeleton(s) for s in skeletons]

        yellows = [s for s, c in skeletons if c == Color.yellow and len(s) > 10]
        whites = [s for s, c in skeletons if c == Color.white and len(s) > 10]

        # Process the yellow line first
        self._process_yellows(yellows)

        # Then move on to the white lines
        self._process_whites(whites)

        if self.verbose and self.intermittent_log_now():
            img = plot_fitted_skeleton([b for b in self.beziers if b.fitted], skeletons)
            img_message = self.bridge.cv2_to_imgmsg(img, 'bgr8')

            self.pub_skeletons_image.publish(img_message)

        self.stats.processed()
        self.intermittent_counter += 1
        if self.intermittent_log_now():
            self.intermittent_log(self.stats.info())
            self.stats.reset()

        message = BeziersMsg()
        message.left = make_bezier_message(self.left)
        message.yellow = make_bezier_message(self.yellow)
        message.right = make_bezier_message(self.right)

        self.pub_bezier.publish(message)

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
