#!/usr/bin/env python
# import cv2
import duckietown_utils as dtu
import rospy
import numpy as np
import torch
import time
from merganser_bezier.bezier import Bezier, BezierLoss, compute_curve
from merganser_msgs.msg import BezierMsg, SkeletonMsg, SkeletonsMsg, BeziersMsg
from duckietown_msgs.msg import Vector2D


class BezierNode(object):
    def __init__(self):
        self.node_name = 'BezierNode'

        self.stats = Stats()
        # Only be verbose every 10 cycles
        self.intermittent_interval = 100
        self.intermittent_counter = 0

        # Parameters
        self.verbose = False
        self.refit_every = 1
        self.test = False
        self.loss_threshold = .1
        self.extension = .1
        self.curve_precision = 100
        self.fitting_steps = 20

        # Subscribers
        self.sub_skeleton = rospy.Subscriber(
            '~skeletons',
            SkeletonsMsg,
            self.process_skeletons,
            queue_size=1
        )

        # Publishers
        self.pub_bezier = rospy.Publisher('~beziers', BeziersMsg, queue_size=1)

        self.update_params(None)

        # Updates the parameters every 2 seconds
        rospy.Timer(rospy.Duration.from_sec(2.), self.update_params)

        # Attributes
        self.steps = 0
        self.beziers = []
        self.loss_function = BezierLoss(1e-2)

        self.time = 0
        self.n = 0

        if self.test:
            self.pub_skeletons = rospy.Publisher('~skeletons', SkeletonsMsg, queue_size=1)
            self.skeletons_timer = rospy.Timer(rospy.Duration.from_sec(.01), self.test_messages)

    def update_params(self, _event):
        # self.loginfo('Updating parameters...')

        self.test = rospy.get_param('~test', False)

        self.verbose = rospy.get_param('~verbose', False)
        self.refit_every = rospy.get_param('~refit_every', 1)
        self.loss_threshold = rospy.get_param('~loss_threshold', .01)
        self.extension = rospy.get_param('~extension', .1)
        self.curve_precision = rospy.get_param('~curve_precision', 100)
        self.fitting_steps = rospy.get_param('~fitting_steps', 20)

        if self.test:
            self.pub_skeletons = rospy.Publisher('~skeletons', SkeletonsMsg, queue_size=1)
            self.skeletons_timer = rospy.Timer(rospy.Duration.from_sec(.01), self.test_messages)
        else:
            self.pub_skeletons = None
            self.skeletons_timer = None

    def test_messages(self, event=None):

        skeletons = SkeletonsMsg()

        controls = np.array([
            [.2, .4],
            [.4, .2],
            [.6, .8],
            [.8, .2],
        ])

        curve = compute_curve(controls)
        cloud = np.random.normal(size=curve.shape) * .1 + curve

        skeleton = SkeletonMsg()
        skeleton.color = skeleton.WHITE

        for i, (x, y) in enumerate(cloud):
            v = Vector2D()
            v.x, v.y = x, y
            skeleton.cloud.append(v)

        skeletons.skeletons.append(skeleton)

        controls = np.array([
            [.4, .2],
            [.2, .4],
            [.8, .6],
            [.2, .8],
        ])

        curve = compute_curve(controls)
        cloud = np.random.normal(size=curve.shape) * .1 + curve

        skeleton = SkeletonMsg()
        skeleton.color = skeleton.WHITE

        for i, (x, y) in enumerate(cloud):
            v = Vector2D()
            v.x, v.y = x, y
            skeleton.cloud.append(v)

        skeletons.skeletons.append(skeleton)

        self.pub_skeletons.publish(skeletons)

    def loginfo(self, message):
        rospy.loginfo('[%s] %s' % (self.node_name, message))

    def _extend_beziers(self):
        for bezier in self.beziers:
            bezier.extrapolate(0 - self.extension, 1 + self.extension)

    def _extract_skeleton(self, skeleton):
        cloud = torch.Tensor([[point.x, point.y] for point in skeleton.cloud])
        color = skeleton.color

        return cloud, color

    def _process_skeleton(self, skeleton):

        # self.loginfo('Getting cloud')

        cloud, color = self._extract_skeleton(skeleton)

        losses = np.array([self.loss_function(b(), cloud) for b in self.beziers])
        argmin = losses.argmin() if len(losses) > 0 else -1

        if argmin > -1 and losses[argmin] < self.loss_threshold:
            # self.loginfo('Re-using curve...')
            bezier = self.beziers[argmin].copy()
        else:
            bezier = Bezier(4, self.curve_precision, cloud)

        # self.loginfo('Loss pre-fit : %.5f' % self.loss_function(bezier(), cloud))

        bezier.fit(
            cloud=cloud,
            loss_function=self.loss_function,
            steps=self.fitting_steps
        )

        # self.loginfo('Loss post-fit : %.5f' % self.loss_function(bezier(), cloud))
        # self.loginfo(' ')

        return bezier, color

    def _make_bezier_message(self, bezier, color):
        msg = BezierMsg()
        msg.color = color
        for i, c in enumerate(bezier.controls.data.numpy()):
            msg.controls[i].x = c[0]
            msg.controls[i].y = c[1]
        return msg

    def process_skeletons(self, skeletons_msg):

        # self.loginfo('Got skeletons')

        t0 = time.time()

        self.stats.received()

        # Extends all collected Bezier curves
        self._extend_beziers()

        # Gets the skeletons
        skeletons = skeletons_msg.skeletons

        # Creates the message containing the updated Bezier curves
        beziers = []
        messages = BeziersMsg()

        for skeleton in skeletons:
            bezier, color = self._process_skeleton(skeleton)
            beziers.append(bezier)

            # Creates the message associated with the bezier curve and appends it to the general message
            msg = self._make_bezier_message(bezier, color)
            messages.beziers.append(msg)

        self.beziers = beziers

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
