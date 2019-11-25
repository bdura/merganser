import numpy as np
import cv2


def detections_to_image(detections):
    B, G, R = 0, 1, 2

    # Add white layer
    image = cv2.cvtColor(detections.white, cv2.COLOR_GRAY2BGR)
    # Add yellow layer
    image[..., G] = cv2.bitwise_or(image[..., G], detections.yellow)
    image[..., R] = cv2.bitwise_or(image[..., R], detections.yellow)
    # Add red layer
    image[..., R] = cv2.bitwise_or(image[..., R], detections.red)

    return image

def skeletons_to_image(skeletons, image_size, step=0):
    B, G, R = 0, 1, 2
    image = np.zeros(image_size, dtype=np.uint8)

    for index, (x, y) in enumerate(skeletons.white):
        image[x, y, :] = max(255 - index * step, 0)
    for index, (x, y) in enumerate(skeletons.yellow):
        image[x, y, G] = max(255 - index * step, 0)
        image[x, y, R] = max(255 - index * step, 0)
    for index, (x, y) in enumerate(skeletons.red):
        image[x, y, R] = max(255 - index * step, 0)

    return image
