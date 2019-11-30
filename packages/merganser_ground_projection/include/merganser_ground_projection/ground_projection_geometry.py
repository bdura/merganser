import itertools

import cv2

import numpy as np
import duckietown_utils as dtu

from duckietown_msgs.msg import Pixel, Vector2D
from geometry_msgs.msg import Point
from image_geometry import PinholeCameraModel


class GroundProjectionGeometry(object):
    def __init__(self, camera_info, homography):
        self.camera_info = camera_info
        self.homography = homography
        self._homography_inv = None

        self.pcm = PinholeCameraModel()
        self.pcm.fromCameraInfo(self.camera_info)

        self._rectify_inited = False
        self._distort_inited = False

    @property
    def homography_inv(self):
        if self._homography_inv is None:
            self._homography_inv = np.linalg.inv(self.homography)
        return self._homography_inv

    # Vector <> Pixel
    def vector_to_pixel(self, vector):
        camera_width = self.camera_info.width
        camera_height = self.camera_info.height
        return Pixel(u=camera_width * vector.x,
                     v=camera_height * vector.y)

    def pixel_to_vector(self, pixel):
        camera_width = self.camera_info.width
        camera_height = self.camera_info.height
        return Vector2D(x=pixel.u / camera_width,
                        y=pixel.v / camera_height)

    # Pixel <> Ground
    def pixel_to_ground(self, pixel):
        uv_raw = np.array([pixel.u, pixel.v, 1])
        x, y, z = np.dot(self.homography, uv_raw)
        return Point(x=x / z, y=y / z, z=0.)

    def ground_to_pixel(self, point):
        if point.z != 0:
            raise ValueError('This method assumes that the point is a ground '
                             'point (z = 0). However, the point is ({0}, {1}, '
                             '{2})'.format(point.x, point.y, point.z))
        ground_point = np.array([point.x, point.y, 1.0])
        x, y, z = np.linalg.solve(self.homography, ground_point)
        return Pixel(u=x / z, v=y / z)

    # Vector <> Ground
    def vector_to_ground(self, vector):
        pixel = self.vector_to_pixel(vector)
        return self.pixel_to_ground(pixel)

    def ground_to_vector(self, point):
        pixel = self.ground_to_pixel(point)
        return self.pixel_to_vector(pixel)

    # Multiple vectors > Ground
    def vectors_to_ground(self, vectors):
        camera_width = self.camera_info.width
        camera_height = self.camera_info.height

        pixels_x = np.asarray([vector.x for vector in vectors]) * camera_width
        pixels_y = np.asarray([vector.y for vector in vectors]) * camera_height
        pixels_z = np.ones_like(pixels_x)

        uv_raws = np.stack((pixels_x, pixels_y, pixels_z), axis=0)
        xs, ys, zs = np.dot(self.homography, uv_raws)

        return [Point(x=x / z, y=y / z, z=0.) for (x, y, z) in zip(xs, ys, zs)]
