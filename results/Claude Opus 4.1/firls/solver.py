import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        """Fast implementation of FIRLS filter design."""
        n, edges = problem
        n = 2 * n + 1  # actual filter length (must be odd)
        
        # JSON round-trip may turn `edges` into a list – convert to tuple
        edges = tuple(edges)
        
        # Use scipy's optimized firls implementation
        coeffs = signal.firls(n, (0.0, *edges, 1.0), [1, 1, 0, 0])
        return coeffs