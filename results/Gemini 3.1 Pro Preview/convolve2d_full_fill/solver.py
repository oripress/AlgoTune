import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        a32 = a.astype(np.float32, copy=False)
        b32 = b.astype(np.float32, copy=False)
        return signal.fftconvolve(a32, b32, mode='full')