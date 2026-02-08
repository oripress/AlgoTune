import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        # correlate2d(a, b, mode='full') == fftconvolve(a, b[::-1, ::-1], mode='full')
        result = fftconvolve(a, b[::-1, ::-1], mode='full')
        return result