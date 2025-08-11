import numpy as np
from scipy import signal

class Solver:
    def __init__(self):
        self.mode = 'full'
    
    def solve(self, problem):
        """Compute 1D correlation for a single pair of arrays."""
        a, b = problem
        return signal.convolve(a, b, mode=self.mode)