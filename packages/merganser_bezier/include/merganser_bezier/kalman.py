import numpy as np
from .utils.kinematics import rotation

from scipy.linalg import block_diag


class KalmanFilter(object):

    def __init__(self, mu, process_noise=.1, measurement_noise=.1):

        self.mu = mu
        self.sigma = np.zeros((8, 8))

        self.r = process_noise * np.eye(8)
        self.q_ = (1 / measurement_noise) * np.eye(2)

    def predict(self, dx, dtheta):
        # Constructs the rotation matrix
        r = block_diag(*[rotation(- dtheta) for _ in range(4)])

        # Constructs the offset
        offset = np.zeros(8)
        offset[list(range(0, 8, 2))] = - dx

        # Computes the new mean
        mu_bar = np.matmul(r, self.mu.reshape(-1) + offset).T

        # Computes the new covariance matrix
        sigma_bar = np.matmul(r, np.matmul(self.sigma, r.T)) + self.r

        self.mu = mu_bar.reshape(4, 2)
        self.sigma = sigma_bar

    def correct(self, bezier, cloud):

        bezier.controls = self.mu

        arg = ((bezier().reshape(1, -1, 2) - cloud.reshape((-1, 1, 2))) ** 2).sum(axis=2).argmin(axis=1)
        b = bezier.bernstein[arg]

        b = np.array([
            np.hstack([b__ * np.eye(2) for b__ in b_])
            for b_ in b
        ])

        btq_ = np.matmul(b.transpose((0, 2, 1)), self.q_)

        sigma_bar_inv = np.linalg.inv(self.sigma)

        sigma_inv = np.matmul(btq_, b).sum(axis=0) + sigma_bar_inv
        sigma = np.linalg.inv(sigma_inv)

        btq_c = np.matmul(btq_, cloud.reshape(-1, 2, 1)).sum(axis=0).reshape(-1)

        mu = np.matmul(
            sigma,
            btq_c + np.matmul(sigma_bar_inv, self.mu.reshape(-1))
        )

        self.mu = mu.reshape(4, 2)
        self.sigma = sigma

    def step(self, dx, dtheta, bezier, cloud):
        self.predict(dx, dtheta)
        self.correct(bezier, cloud)
