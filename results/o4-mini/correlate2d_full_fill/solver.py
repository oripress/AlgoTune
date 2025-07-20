import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Compute the 2D correlation of arrays a and b using 'full' mode
        and 'fill' boundary via FFT-based convolution with float32 precision.
        """
        a, b = problem
        # Cast inputs to float32 for faster FFT operations
        a_f = np.asarray(a, dtype=np.float32)
        b_f = np.asarray(b, dtype=np.float32)
        # Correlation is convolution with reversed kernel
        c_f = fftconvolve(a_f, b_f[::-1, ::-1], mode='full')
        # Cast back to float64 for output compatibility
        return c_f.astype(np.float64)