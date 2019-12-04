import math
import numpy as np

from collections import namedtuple


Velocities = namedtuple('Velocities', 'v omega')

def follow_point_to_velocities(follow_point,
                               previous,
                               v_max=1.0,
                               decay_rate=0.9,
                               gain_min_velocity=0.4,
                               gain_omega=0.5,
                               sin_alpha_min=0.1):
    if follow_point is not None:
        # Compute the distance to the follow point. This should
        # almost always be equal to look_ahead_distance
        L = math.sqrt(follow_point[0] ** 2 + follow_point[1] ** 2)

        # Compute sin(alpha)
        sin_alpha = follow_point[1] / L

        # Compute the velocity (inversely propotional to sin(alpha))
        if np.isclose(sin_alpha, 0.):
            v_bar = v_max
        else:
            v_bar = v_max * min(1., sin_alpha_min / sin_alpha)
            v_bar = max(v_bar, gain_min_velocity * v_max)
        # Smooth the velocity
        v = decay_rate * previous.v + (1. - decay_rate) * v_bar

        # Compute the angular velocity
        omega_bar = 2. * gain_omega * v * sin_alpha / L

    else:
        # If no follow point (so line detected), then fall back to default 
        # behavior (moving straight at maximum velocity)
        v = decay_rate * previous.v + (1. - decay_rate) * v_max
        omega_bar = 0.

    omega = decay_rate * previous.omega + (1. - decay_rate) * omega_bar

    return Velocities(v=v, omega=omega)
