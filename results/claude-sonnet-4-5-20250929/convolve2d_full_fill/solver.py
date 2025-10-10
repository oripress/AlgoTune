import numpy as np
from scipy import signal

class Solver:
    def __init__(self):
        self.mode = 'full'
        self.boundary = 'fill'
    
    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using "full" mode and "fill" boundary.

        :param problem: A tuple (a, b) of 2D arrays.
        :return: A 2D array containing the convolution result.
        """
        a, b = problem
        # Convert to float32 for faster computation
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        result = signal.fftconvolve(a, b, mode=self.mode)
        return result