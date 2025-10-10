import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: tuple, **kwargs) -> np.ndarray:
        """Compute 1D convolution for a single pair of arrays."""
        a, b = problem
        a = np.asarray(a)
        b = np.asarray(b)
        
        # Use FFT for larger arrays, direct for smaller
        n, m = len(a), len(b)
        if n * m > 500:
            return signal.fftconvolve(a, b, mode='full')
        else:
            return signal.convolve(a, b, mode='full')