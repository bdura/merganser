import numpy as np

from merganser_msgs.msg import SkeletonMsg, SkeletonsMsg
from duckietown_msgs.msg import Vector2D


def skeleton_to_msg(skeleton, color, original_image_size, top_cutoff=0):
    x_indices, y_indices = skeleton

    x_float = x_indices.astype(np.float32)
    y_float = y_indices.astype(np.float32)

    # Normalize the indices in [0, 1]
    x_float = 1 - (x_float + top_cutoff) / (original_image_size[0] - 1)
    y_float = 1 - y_float / (original_image_size[1] - 1)

    cloud = []
    for y, x in zip(y_float, x_float):
        cloud.append(Vector2D(x, y))

    # Create message
    message = SkeletonMsg()
    message.cloud = cloud
    message.color = color

    return message


def skeletons_to_msg(skeletons, original_image_size, top_cutoff=0):
    messages = []
    # Add all the white skeletons
    for skeleton in skeletons.white:
        messages.append(skeleton_to_msg(skeleton, SkeletonMsg.WHITE,
                        original_image_size, top_cutoff=top_cutoff))
    # Add all the yellow skeletons
    for skeleton in skeletons.yellow:
        messages.append(skeleton_to_msg(skeleton, SkeletonMsg.YELLOW,
                        original_image_size, top_cutoff=top_cutoff))
    # Add all the red skeletons
    for skeleton in skeletons.red:
        messages.append(skeleton_to_msg(skeleton, SkeletonMsg.RED,
                        original_image_size, top_cutoff=top_cutoff))

    message = SkeletonsMsg()
    message.skeletons = messages

    return message
