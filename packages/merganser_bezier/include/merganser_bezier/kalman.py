import numpy as np
from scipy.linalg import block_diag

from .utils.bernstein import get_bernstein
from .utils.kinematics import rotation


class KalmanFilter(object):

    def __init__(self, dimension=4, process_noise=.1):
        self.dimension = dimension

        self.mu = np.empty(dimension * 2)
        self.sigma = np.zeros((dimension * 2, dimension * 2))
        self.r = process_noise * np.eye(dimension * 2)

    def reset(self, mu):
        """
        Resets the covariance matrix to zero (after a gradient descent step for example).
        """

        self.mu = mu.reshape(-1)
        self.sigma = np.zeros((self.dimension * 2, self.dimension * 2))

    def predict(self, dx, dtheta):
        r"""
        Updates the mean and covariance of the estimate based on the control inputs
        (computes :math:`\bar{bel}(\theta_t)`.

        Parameters
        ----------
        dx: float
            The distance run by the robot in the interval.
        dtheta: float
            The rotation performed by the robot in the time step.
        """

        # Constructs the rotation matrix
        r = block_diag(*[rotation(- dtheta) for _ in range(self.dimension)])

        # Constructs the offset
        offset = np.zeros(self.dimension * 2)
        offset[list(range(0, self.dimension * 2, 2))] = - dx

        # Computes the new mean
        mu_bar = np.matmul(r, self.mu + offset).T

        # Computes the new covariance matrix
        sigma_bar = np.matmul(r, np.matmul(self.sigma, r.T)) + self.r

        self.mu = mu_bar
        self.sigma = sigma_bar

    def correct(self, cloud):
        """
        Corrects the mean and covariance of the estimate using the measurement.

        Parameters
        ----------
        cloud: np.array
            The measured cloud point.
        """

        bernstein = get_bernstein(order=self.dimension)

        curve = np.matmul(bernstein, self.mu.reshape((self.dimension, 2)))

        arg = ((curve.reshape((1, -1, 2)) - cloud.reshape((-1, 1, 2))) ** 2).sum(axis=2).argmin(axis=1)
        b = bernstein[arg]

        cov = np.cov((curve[arg] - cloud).T)

        b = np.array([
            np.hstack([b__ * np.eye(2) for b__ in b_])
            for b_ in b
        ])

        btq_ = np.matmul(b.transpose((0, 2, 1)), np.linalg.inv(cov))

        sigma_bar_inv = np.linalg.inv(self.sigma)

        sigma_inv = np.matmul(btq_, b).sum(axis=0) + sigma_bar_inv
        sigma = np.linalg.inv(sigma_inv)

        btq_c = np.matmul(btq_, cloud.reshape(-1, 2, 1)).sum(axis=0).reshape(-1)

        mu = np.matmul(
            sigma,
            btq_c + np.matmul(sigma_bar_inv, self.mu)
        )

        self.mu = mu
        self.sigma = sigma

    def fit(self, dx, dtheta, cloud):
        """
        Performs a single step of predict/correct step, using the control inputs and the measured cloud.

        Parameters
        ----------
        dx: float
            The distance run by the robot in the interval.
        dtheta: float
            The rotation performed by the robot in the time step.
        cloud: np.array
            The measured cloud point.
        """
        self.predict(dx, dtheta)
        self.correct(cloud)
