import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _proj_sorted_numba(y, sorted_y):
    n = y.shape[0]
    # Compute threshold theta by scanning sorted ascending array from largest to smallest
    sum_y = 0.0
    sum_y_at_rho = 0.0
    rho = -1
    for idx in range(n - 1, -1, -1):
        yi = sorted_y[idx]
        sum_y += yi
        k = n - idx
        t = (sum_y - 1.0) / k
        if yi > t:
            rho = k
            sum_y_at_rho = sum_y
    theta = (sum_y_at_rho - 1.0) / rho
    # Allocate output and project
    x = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = y[i] - theta
        x[i] = v if v > 0.0 else 0.0
    return x

# Trigger JIT compilation on small arrays
_proj_sorted_numba(np.empty(1, dtype=np.float64), np.empty(1, dtype=np.float64))

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """Project vector y onto the probability simplex."""
        y = np.asarray(problem.get("y"), dtype=np.float64)
        # sort ascending outside Numba
        sorted_y = np.sort(y)
        # compute projection via JIT-compiled loops
        x = _proj_sorted_numba(y, sorted_y)
        return {"solution": x}