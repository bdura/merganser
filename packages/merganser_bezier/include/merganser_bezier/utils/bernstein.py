import autograd.numpy as np
from scipy.special import comb


def memoize(func):
    memory = dict()

    def f(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in memory:
            memory[key] = func(*args, **kwargs)
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
