import numpy as np
from scipy import signal

class Solver:
    def __init__(self):
        self.mode = 'full'  # Default mode
    
    def solve(self, problem, **kwargs):
        """Compute 1D correlation for pairs of arrays."""
        a, b = problem
        # Use FFT-based convolution which is faster for larger arrays
        return signal.fftconvolve(a, b, mode=self.mode)