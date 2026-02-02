import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        # Convert to float32 for faster computation
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        b32 = np.ascontiguousarray(b[::-1, ::-1], dtype=np.float32)
        result = fftconvolve(a32, b32, mode='full')
        return result