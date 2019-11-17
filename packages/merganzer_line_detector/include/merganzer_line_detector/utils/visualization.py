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
