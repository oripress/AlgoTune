import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs) -> any:
        # Handle single pair case
        if len(problem) == 2 and hasattr(problem[0], '__len__') and hasattr(problem[1], '__len__'):
            return signal.fftconvolve(problem[0], problem[1], mode='full')
        
        # Handle list of pairs case
        return [signal.fftconvolve(a, b, mode='full') for a, b in problem]