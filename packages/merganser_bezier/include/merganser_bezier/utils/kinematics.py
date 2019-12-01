import numpy as np


def rotation(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    r = np.array([[cos, -sin],
                  [sin, cos]])
    return r


def se3(v, omega, dt):

    m = np.zeros((3, 3))
    m[:2, :2] = rotation(omega * dt)
    m[-1, -1] = 1
    m[0, -1] = v * dt

    return m


def transform(v, omega, dt, points):
    points_ = np.ones((len(points), 3))
    points_[:, :2] = points

    transformation = se3(v, omega, dt)

    return np.matmul(transformation, points_.T)[:2].T
