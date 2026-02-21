import numpy as np
from scipy import signal

class Solver:
    def __init__(self, mode="full"):
        self.mode = mode

    def solve(self, problem, **kwargs):
        a, b = problem
        return signal.convolve(a, b, mode=self.mode)