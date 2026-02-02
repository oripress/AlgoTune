import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        return signal.fftconvolve(a, b, mode='full')