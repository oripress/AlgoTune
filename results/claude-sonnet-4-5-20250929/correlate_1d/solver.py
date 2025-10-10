import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def __init__(self, mode="full"):
        """Initialize solver with correlation mode."""
        self.mode = mode
    
    def solve(self, problem: list) -> list:
        """
        Compute the 1D correlation for each valid pair in the problem list.
        
        For mode 'valid', process only pairs where the length of the second array does not exceed the first.
        Return a list of 1D arrays representing the correlation results.
        
        :param problem: A list of tuples of 1D arrays.
        :return: A list of 1D correlation results.
        """
        results = []
        
        for a, b in problem:
            if self.mode == "valid" and b.shape[0] > a.shape[0]:
                continue
            
            # Use fftconvolve with reversed b for correlation
            res = fftconvolve(a, b[::-1], mode=self.mode)
            results.append(res)
        return results