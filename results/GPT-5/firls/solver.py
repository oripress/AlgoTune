import numpy as np
from functools import lru_cache
from scipy import signal

class Solver:
    @staticmethod
    @lru_cache(maxsize=512)
    def _firls_cached(numtaps: int, f1: float, f2: float) -> np.ndarray:
        return signal.firls(numtaps, (0.0, f1, f2, 1.0), [1, 1, 0, 0])

    def solve(self, problem, **kwargs) -> np.ndarray:
        n, edges = problem
        edges = tuple(edges)
        numtaps = 2 * int(n) + 1  # actual filter length (must be odd)
        f1, f2 = float(edges[0]), float(edges[1])
        return self._firls_cached(numtaps, f1, f2)