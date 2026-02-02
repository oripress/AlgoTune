import numpy as np
from scipy import signal
try:
    from convolution import solve_cython
except ImportError:
    solve_cython = None

class Solver:
    def __init__(self):
        self.mode = 'full'

    def solve(self, problem, **kwargs):
        if solve_cython is not None and isinstance(problem, list):
            return solve_cython(problem)
            
        # Fallback
        if isinstance(problem, list):
            results = []
            for p in problem:
                a, b = p
                a = np.asarray(a)
                b = np.asarray(b)
                results.append(signal.convolve(a, b, mode=self.mode))
            return results
        else:
            a, b = problem
            return signal.convolve(a, b, mode=self.mode)