import numpy as np
from scipy.signal import firls

class Solver:
    def solve(self, problem, **kwargs):
        """
        FIR filter design using least-squares (firls).
        problem: (M, (f1, f2)) where
          M = half-filter length (so total length N = 2*M + 1),
          f1, f2 are normalized band edges (0 to 1).
        """
        M, edges = problem
        M = int(M)
        # actual number of taps (odd)
        n = 2 * M + 1
        # ensure edges tuple
        f1, f2 = edges
        return firls(n, (0.0, f1, f2, 1.0), [1, 1, 0, 0])