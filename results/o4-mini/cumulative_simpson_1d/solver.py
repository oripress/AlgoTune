import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _cum_simp(y, dx):
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n < 2:
        for i in range(n):
            out[i] = 0.0
        return out
    out[0] = 0.0
    out[1] = 0.5 * (y[0] + y[1]) * dx
    sum_odd = 0.0
    sum_even = 0.0
    # process in pairs to avoid per-iteration branches
    for k in range(2, n, 2):
        idx = k - 1
        sum_odd += y[idx]
        # composite Simpson for even segment
        out[k] = (y[0] + y[k] + 4.0 * sum_odd + 2.0 * sum_even) * dx / 3.0
        if k + 1 < n:
            # trapezoidal for odd segment
            sum_even += y[k]
            out[k + 1] = out[k] + 0.5 * (y[k] + y[k + 1]) * dx
    return out

# trigger compilation at import time
_cum_simp(np.zeros(2, dtype=np.float64), np.float64(1.0))

class Solver:
    def solve(self, problem, **kwargs):
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        n = y.shape[0]
        if n < 2:
            return np.empty(0, dtype=np.float64)
        # compute cumulative Simpson and drop the initial zero
        return _cum_simp(y, dx)[1:]