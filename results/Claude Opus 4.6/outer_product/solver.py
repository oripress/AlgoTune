import numpy as np

_outer = np.outer

class Solver:
    def solve(self, problem, **kwargs):
        v1, v2 = problem
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        return np.float64(1.0) * a[:, None] * b[None, :]