import numpy as np
from enum import Enum


class State(Enum):

    Straight = 0
    Left = 1
    Right = -1


class FiniteStateMachine(object):

    def __init__(self, measurement_noise=.01):

        self.likelihood = np.ones(3) / 3

    def correct(self, left, yellow, right):

        pass

