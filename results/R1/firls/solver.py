import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        n, edges = problem
        n = 2 * n + 1  # actual filter length (must be odd)
        edges = tuple(edges)  # Ensure edges is a tuple
        coeffs = signal.firls(n, (0.0, *edges, 1.0), [1, 1, 0, 0])
        return coeffs