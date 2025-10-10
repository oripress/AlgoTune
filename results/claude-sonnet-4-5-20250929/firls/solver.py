import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        numtaps = 2 * n + 1
        edges = tuple(edges)
        
        # Use scipy's implementation since it's already optimized
        # and we need to match its output exactly
        coeffs = signal.firls(numtaps, (0.0, *edges, 1.0), [1, 1, 0, 0])
        return coeffs