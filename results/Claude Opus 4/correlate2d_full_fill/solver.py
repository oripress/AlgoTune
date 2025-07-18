import numpy as np
from scipy import signal

class Solver:
    def __init__(self):
        self.mode = 'full'
        self.boundary = 'fill'
    
    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2D correlation of arrays a and b using "full" mode and "fill" boundary.
        
        :param problem: A tuple (a, b) of 2D arrays.
        :return: A 2D array containing the correlation result.
        """
        a, b = problem
        
        # For correlation, we need to flip b (conjugate for complex, flip for real)
        b_flipped = np.flip(b)
        
        # Use FFT-based convolution which is much faster for larger arrays
        result = signal.fftconvolve(a, b_flipped, mode=self.mode)
        
        return result