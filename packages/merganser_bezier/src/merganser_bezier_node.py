#!/usr/bin/env python
# import cv2
import duckietown_utils as dtu
import rospy
import numpy as np
import torch
from merganser_bezier.bezier import Bezier, BezierLoss, compute_curve
from merganser_msgs.msg import BezierMsg, SkeletonMsg, SkeletonsMsg, BeziersMsg
from duckietown_msgs.msg import Vector2D


class BezierNode(object):
    def __init__(self):
        self.node_name = 'BezierNode'

        # Parameters
        self.verbose = False
        self.refit = 15

        # Subscribers
        self.sub_skeleton = rospy.Subscriber(
            '~skeletons',
            SkeletonsMsg,
            self.process_skeletons,
            queue_size=1
        )
        self.steps = 0

        # Publishers
        self.pub_bezier = rospy.Publisher('~beziers', BeziersMsg, queue_size=1)

        self.update_params(None)

        rospy.Timer(rospy.Duration.from_sec(2.), self.update_params)

    def loginfo(self, message):
        rospy.loginfo('[%s] %s' % (self.node_name, message))

    def update_params(self, _event):
        self.loginfo('Updating...')
        self.verbose = rospy.get_param('~verbose', False)
        self.refit = rospy.get_param('~refit', 1)

    def _process_skeleton(self, skeleton):

        self.loginfo('Getting cloud')

        cloud = torch.Tensor([[point.x, point.y] for point in skeleton.cloud])
        color = skeleton.color

        bezier = Bezier(4, 20, cloud)
        loss_function = BezierLoss(1e-2)

        self.loginfo('Loss pre-fit : %.2f' % loss_function(bezier(), cloud))

        bezier.fit(
            cloud=cloud,
            loss_function=loss_function,
            steps=20
        )

        self.loginfo('Loss post-fit : %.2f' % loss_function(bezier(), cloud))

        b = BezierMsg()
        b.color = color
        for i, c in enumerate(bezier.controls.data.numpy()):
            b.controls[i].x = c[0]
            b.controls[i].y = c[1]

        return b

    def process_skeletons(self, skeletons_msg):
        skeletons = skeletons_msg.skeletons
        beziers = BeziersMsg()

        for skeleton in skeletons:
            beziers.beziers.append(self._process_skeleton(skeleton))

        self.pub_bezier.publish(beziers)

    def on_shutdown(self):
        self.loginfo('Shutdown...')


if __name__ == '__main__':
    rospy.init_node('merganser_bezier_node', anonymous=False)
    bezier_node = BezierNode()
    rospy.on_shutdown(bezier_node.on_shutdown)
    rospy.spin()
