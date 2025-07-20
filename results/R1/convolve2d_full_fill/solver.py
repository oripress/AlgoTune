import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        
        try:
            # Use parallel workers if available (scipy >= 1.6.0)
            return fftconvolve(a, b, mode='full', workers=-1)
        except TypeError:
            # Fallback for older SciPy versions without workers parameter
            return fftconvolve(a, b, mode='full')