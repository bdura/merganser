import numpy as np
import torch
from scipy.special import comb
from torch import nn


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

    return b @ controls


def compute_curve(controls):
    ts = np.linspace(0, 1, 100)
    b = np.asarray([bernstein(t, len(controls)) for t in ts])
    return b @ controls


class Bezier(nn.Module):

    def __init__(self, order, precision, choice=None):
        super(Bezier, self).__init__()

        ts = np.linspace(0, 1, precision)

        self.bernstein = torch.Tensor([bernstein(t, order) for t in ts])

        if choice is None:
            controls = torch.randn((order, 2))
        else:
            choice = torch.Tensor(choice)
            permutation = torch.randperm(choice.size(0))
            indices = permutation[:order]

            controls = choice[indices]

            arg = controls[:, 0].argsort()
            controls = controls[arg]
        self.controls = nn.Parameter(data=controls)

    def forward(self):
        return self.bernstein @ self.controls

    def extrapolate(self, t0, t1):
        controls = self.controls.data.numpy()
        self.controls.data = torch.Tensor(extrapolate(controls, t0, t1))


class BezierLoss(object):

    def __init__(self, alpha=1e-3):
        """
        Initialises the object.

        Parameters
        ----------
        alpha: float
            Governs the regularisation on the extreme points.
            With `alpha = 0`, the curve can be longer than necessary and have irrelevant extremities.
        """

        self.alpha = alpha

    def __call__(self, curve, cloud):
        r"""
        Computes the "_Bezier loss_" between a Bezier curve and a cloud of points to be fitted.

        The computed loss is :
        .. math::

            \mathcal{L}(B, C) = \frac{1}{|C|} \sum_{c \in C} \min_{b \in B} ||c - b||^2

        We add a regularization term to make sure that the computed curve
        is not longer than necessary (which could lead to incorrect deductions),
        by penalising the distance between the extremities of the curve and the cloud.

        Parameters
        ----------
        curve: torch.Tensor
            The Bezier curve to evaluate.
        cloud: torch.Tensor
            The cloud of points to be fitted.

        Returns
        -------
        loss: torch.Tensor
            The goodness-of-fit loss between the curve and the point cloud.
        """

        curve = curve.unsqueeze(0)
        cloud = cloud.unsqueeze(1)

        se = ((curve - cloud) ** 2).mean(dim=-1)
        loss = se.min(dim=1)[0].mean()

        loss = loss + self.alpha * se[:, [0, -1]].min(dim=0)[0].mean()

        return loss


def decasteljau(controls, t, left=None, right=None):
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

        new_anchors = (1-t) * left_anchors + t * right_anchors

        if memory:
            left.append(controls[0])
            right.append(controls[-1])

        return decasteljau(new_anchors, t, left, right)


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
    decasteljau(controls, t, left, right)
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
