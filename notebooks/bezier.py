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
    return np.array([compute(t, controls) for t in ts])


class Bezier(nn.Module):

    def __init__(self, order, precision, choice=None):
        super(Bezier, self).__init__()

        ts = np.linspace(0, 1, precision)

        self.bernstein = torch.Tensor([bernstein(t, order) for t in ts])

        if choice is None:
            controls = torch.randn((order, 2))
        else:
            permutation = torch.randperm(choice.size(0))
            indices = permutation[:order]

            controls = choice[indices]

            arg = controls[:, 0].argsort()
            controls = controls[arg]
        self.controls = nn.Parameter(data=controls)

    def forward(self):
        return self.bernstein @ self.controls


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

        loss = loss + se[:, [0, -1]].min(dim=0)[0].mean() * self.alpha

        return loss
