from copy import deepcopy

import autograd.numpy as np
from autograd import grad

from .utils.optimisers import adam
from .utils.bernstein import get_bernstein, compute_curve, extrapolate

from .kalman import KalmanFilter


COLORS = {
    0: 'white',
    1: 'yellow',
    2: 'red',
    -1: 'gray'
}


class Bezier(object):

    def __init__(self, order, precision, reg=5e-3, process_noise=.01, loss_threshold=.001, color=-1):
        super(Bezier, self).__init__()

        self.bernstein = get_bernstein(precision=precision, order=order)

        self.controls = np.empty((order, 2))

        self.cloud = None
        self.reg = reg

        self.loss_threshold = loss_threshold
        self.filter = KalmanFilter(dimension=order, process_noise=process_noise)

        self.color = COLORS[color]

    def initialise(self, cloud):

        order = len(self.controls)

        if cloud is None:
            self.controls = np.random.normal(size=(order, 2))
        else:

            n = len(cloud) // order + (len(cloud) % order > 0)

            x_args = cloud[:, 0].argsort()
            x_controls = np.array([
                cloud[x_args[i * n:(i + 1) * n]].mean(axis=0)
                for i in range(order)]
            )

            self.controls = x_controls
            x_loss = self.loss(cloud)

            y_args = cloud[:, 1].argsort()
            y_controls = np.array([
                cloud[y_args[i * n:(i + 1) * n]].mean(axis=0)
                for i in range(order)]
            )

            self.controls = y_controls
            y_loss = self.loss(cloud)

            if y_loss > x_loss:
                self.controls = x_controls

            self.extrapolate(-.1, 1.1)

    def __call__(self):
        return np.matmul(self.bernstein, self.controls)

    @classmethod
    def from_controls(cls, controls, precision=10, **kwargs):
        bezier = Bezier(4, precision, **kwargs)
        bezier.controls = controls
        return bezier

    def squared_error(self, cloud):
        curve = self()
        diff = curve.reshape(-1, 1, 2) - cloud.reshape(1, -1, 2)
        se = (diff ** 2).sum(axis=2)
        return se

    def nll(self, cloud):
        se = self.squared_error(cloud)
        return se.min(axis=0).sum()

    def closed_form(self, cloud):
        se = self.squared_error(cloud)

        argmin = se.argmin(axis=0)

        b = self.bernstein[argmin]

        btb_inv = np.linalg.inv(np.matmul(b.T, b))
        btb_inv_bt = np.matmul(btb_inv, b.T)

        self.controls = np.matmul(btb_inv_bt, cloud)

    def objective(self, params, cloud):
        curve = np.matmul(self.bernstein, params)
        diff = curve.reshape(-1, 1, 2) - cloud.reshape(1, -1, 2)
        se = (diff ** 2).mean(axis=2)

        return se.min(axis=0).mean() + self.reg * se[[0, -1]].min(axis=1).mean()

    def create_objective(self, cloud):

        def obj(params, *args):
            return self.objective(params, cloud)

        return obj

    def loss(self, cloud):
        return self.objective(self.controls, cloud)

    def extrapolate(self, t0, t1):
        self.controls = extrapolate(self.controls, t0, t1)

    def fit(self, cloud, steps=20, lr=.1, eps=1e-3):
        objective = self.create_objective(cloud)
        gradient = grad(objective)
        self.controls = adam(gradient, self.controls, step_size=lr, num_iters=steps, threshold=eps)

    def collapse(self, cloud):
        self.initialise(cloud)
        self.fit(
            cloud=cloud,
            steps=200,
            eps=.001,
            lr=.005,
        )
        self.filter.reset(self.controls)

    def kalman(self, dx, dtheta, cloud):
        self.filter.fit(dx, dtheta, cloud)
        self.controls = self.filter.mu.reshape((len(self.controls), 2))

    def predict(self, dx, dtheta):
        self.filter.predict(dx, dtheta)
        self.controls = self.filter.mu.reshape((len(self.controls), 2))

    def correct(self, cloud):
        self.filter.correct(cloud)
        self.controls = self.filter.mu.reshape((len(self.controls), 2))

        if self.loss(cloud) > self.loss_threshold:
            self.collapse(cloud)

    def step(self, dx, dtheta, cloud):

        self.kalman(dx, dtheta, cloud)

        if self.loss(cloud) > self.loss_threshold:
            self.collapse(cloud)

    def derivative(self):
        n = len(self.controls)
        c0, cn = self.controls[:-1], self.controls[1:]

        c = n * (cn - c0)
        d = compute_curve(c, len(self.bernstein))

        return d

    def normal(self):
        # Gets the derivative
        d = self.derivative()

        # Normalises the derivative
        d = d / np.linalg.norm(d, axis=1, keepdims=True)

        rot = np.array([[0, 1], [-1, 0]])

        return np.matmul(rot, d.T).T

    def copy(self):
        new = deepcopy(self)
        return new
