import numpy as np
from typing import Any
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Use scipy.signal.firls to reproduce the reference FIR least-squares design.
        The input `n` is interpreted as in the reference: actual filter length =
        2*n + 1 (odd length).
        problem: (n, (f1, f2))
        returns: 1D numpy array of filter coefficients (length 2*n + 1)
        """
        if not isinstance(problem, (tuple, list)) or len(problem) < 2:
            raise ValueError("problem must be a tuple (n, (f1, f2))")

        n, edges = problem
        n = int(n)
        numtaps = 2 * n + 1  # match reference's interpretation

        edges = tuple(edges)
        if len(edges) != 2:
            raise ValueError("edges must be a pair (f1, f2)")

        f1, f2 = float(edges[0]), float(edges[1])
        bands = (0.0, f1, f2, 1.0)
        desired = [1, 1, 0, 0]
        coeffs = signal.firls(numtaps, bands, desired)
        return np.asarray(coeffs, dtype=np.float64)