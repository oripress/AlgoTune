import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the 2D convolution using FFT-based approach with single precision.
        """
        a, b = problem
        # Convert to single precision for faster FFT
        a32 = a.astype(np.float32)
        b32 = b.astype(np.float32)
        result = signal.fftconvolve(a32, b32, mode='full')
        return result.astype(np.float64)