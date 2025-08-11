import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Compute the 2D convolution of two arrays using FFT-based convolution.
        This matches scipy.signal.convolve2d with mode='full' and boundary='fill'.
        """
        a, b = problem
        # Use FFT-based convolution for speed.
        result = fftconvolve(a, b, mode='full')
        # Ensure real output (small imaginary parts may appear due to numerical errors)
        if np.iscomplexobj(result):
            result = np.real_if_close(result, tol=1e-12)
        return result