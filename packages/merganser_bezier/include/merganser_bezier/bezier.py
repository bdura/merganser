from copy import deepcopy

import autograd.numpy as np
from autograd import grad
from scipy.special import comb

from .utils.optimisers import adam


def memoize(func):
    memory = dict()

    def f(*args):
        key = str(args)
        if key not in memory:
            memory[key] = func(*args)
        return memory[key]

    return f


def bernstein(t, n):
    """
    Computes the Bernstein coefficient of a `n`-th order Bezier curve for a given `t`.

    Parameters
    ----------
    t: float
        The ratio to evaluate.
    n: int
        The order of the Bezier curve.

    Returns
    -------
    b: np.array
        A `n`-dimensional array.
    """
    return np.array([
        comb(n - 1, i) * ((1 - t) ** (n - 1 - i)) * (t ** i)
        for i in range(n)
    ])


def compute(t, controls):
    n = len(controls)
    b = bernstein(t, n)

    return np.matmul(b, controls)


@memoize
def get_bernstein(precision=100, order=4):
    ts = np.linspace(0, 1, precision)
    return np.asarray([bernstein(t, order) for t in ts])


def compute_curve(controls, n=100):
    order = len(controls)
    b = get_bernstein(n, order)
    return np.matmul(b, controls)


class Bezier(object):

    def __init__(self, order, precision, reg=5e-3, choice=None):
        super(Bezier, self).__init__()

        ts = np.linspace(0, 1, precision)
        self.bernstein = np.array([bernstein(t, order) for t in ts])

        if choice is None or len(choice) < order:
            controls = np.random.normal(size=(order, 2))
        else:
            x_argmin, x_argmax = choice[:, 0].argmin(), choice[:, 0].argmax()
            y_argmin, y_argmax = choice[:, 1].argmin(), choice[:, 1].argmax()

            x_norm = ((choice[x_argmax] - choice[x_argmin]) ** 2).sum()
            y_norm = ((choice[y_argmax] - choice[y_argmin]) ** 2).sum()

            if x_norm > y_norm:
                argmin, argmax = x_argmin, x_argmax
            else:
                argmin, argmax = y_argmin, y_argmax

            controls = choice[[argmin]] + (choice[[argmax]] - choice[[argmin]]) * np.linspace(0, 1, 4).reshape(-1, 1)

        self.controls = controls

        self.cloud = None
        self.reg = reg

    @classmethod
    def from_controls(cls, controls, precision=10):
        bezier = Bezier(4, precision)
        bezier.controls = controls
        return bezier

    def __call__(self):
        return np.matmul(self.bernstein, self.controls)

    def nll(self, cloud):
        curve = self()
        diff = curve.reshape(-1, 1, 2) - cloud.reshape(1, -1, 2)
        se = (diff ** 2).sum(axis=2)
        return se.min(axis=0).sum()

    def closed_form(self, cloud):
        curve = self()
        diff = curve.reshape(-1, 1, 2) - cloud.reshape(1, -1, 2)
        se = (diff ** 2).mean(axis=2)

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

    def derivative(self):
        n = len(self.controls)
        c0, cn = self.controls[:-1], self.controls[1:]

        c = n * (cn - c0)
        d = compute_curve(c, len(self.bernstein), order=3)

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


def de_casteljau(controls, t, left=None, right=None):
    """
    Implements de Casteljau's algorithm to obtain the point at parameter value `t`.

    It also provides the control points for the split Bezier curve
    (if provided the two corresponding lists, which are populated).

    Parameters
    ----------
    controls: np.array
        The control points of the Bezier curve.
    t: float
        The parameter value to compute/on which to split.
    left: list, optional
        A python list to be populated with the new control points.
    right: list, optional
        A python list to be populated with the new control points.

    Returns
    -------
    np.array
        The point of the curve corresponding to parameter value t.
    """
    n = len(controls)

    memory = left is not None and right is not None

    if n == 1:
        anchor = controls[0]
        if memory:
            left.append(anchor)
            right.append(anchor)
        return anchor
    else:
        left_anchors = controls[:n - 1]
        right_anchors = controls[1:]

        new_anchors = (1 - t) * left_anchors + t * right_anchors

        if memory:
            left.append(controls[0])
            right.append(controls[-1])

        return de_casteljau(new_anchors, t, left, right)


def split(controls, t):
    """
    Splits a Bezier curve around the parameter value t.

    Parameters
    ----------
    controls: np.array
        The control points of the Bezier curve.
    t: float
        The parameter value to compute/on which to split.

    Returns
    -------
    tuple(np.array)
        The new control points.
    """
    left, right = [], []
    de_casteljau(controls, t, left, right)
    return np.asarray(left), np.asarray(right)


def extrapolate(controls, t0, t1):
    """
    Computes the new control points that "extrapolate" the Bezier curve from `t0` to `t1`.

    Parameters
    ----------
    controls: np.array
        The control points of the Bezier curve.
    t0, t1: float
        The parameter values to extrapolate between.

    Returns
    -------
    new: np.array
        The new control points.
    """
    assert t0 < t1
    left, right = split(controls, t0)
    new, _ = split(right[::-1], t1)
    return new
