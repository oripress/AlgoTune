import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Computes 2D correlation using SciPy's fftconvolve for performance.
        
        The method leverages the equivalence between correlation and convolution
        with a flipped kernel: correlate(a, b) = convolve(a, flip(b)).
        
        `scipy.signal.fftconvolve` uses the Fast Fourier Transform (FFT) to
        perform the convolution. This is significantly faster for large
        arrays than direct computation and is a robust, high-performance
        alternative to other libraries that were causing linter issues.
        """
        a, b = problem
        
        # For correlation, we convolve with a kernel that is flipped along both axes.
        # np.flip is efficient for this operation.
        b_flipped = np.flip(b)
        
        # `fftconvolve` computes the convolution. The 'full' mode is the default
        # and corresponds to the standard definition of correlation.
        # The output is a float64 numpy array, matching the required format.
        return fftconvolve(a, b_flipped, mode='full')