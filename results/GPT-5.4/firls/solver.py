from __future__ import annotations

from functools import lru_cache

from scipy import signal

@lru_cache(maxsize=256)
def _design(n: int, fp: float, fs: float):
    return signal.firls(2 * n + 1, (0.0, fp, fs, 1.0), [1, 1, 0, 0])

class Solver:
    def solve(self, problem, **kwargs):
        n, edges = problem
        fp, fs = map(float, edges)
        return _design(int(n), fp, fs).copy()