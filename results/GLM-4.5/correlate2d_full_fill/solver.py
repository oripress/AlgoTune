import numpy as np
from scipy import signal

class Solver:
    def __init__(self):
        self.mode = 'full'
        self.boundary = 'fill'
    
    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Compute the 2D correlation of arrays a and b using "full" mode and "fill" boundary.

        :param problem: A tuple (a, b) of 2D arrays.
        :return: A 2D array containing the correlation result.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        a, b = problem
        
        # Use scipy's fftconvolve which is often faster than correlate2d
        # For correlation, we need to flip the kernel
        b_flipped = np.flip(b)
        result = signal.fftconvolve(a, b_flipped, mode='full')
        
        return result