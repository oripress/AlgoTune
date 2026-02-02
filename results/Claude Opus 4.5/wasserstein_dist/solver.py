import numpy as np
from numba import njit

@njit('float64(float64[::1], float64[::1])', fastmath=True, cache=True)
def wasserstein_1d(u, v):
    n = u.shape[0]
    if n <= 1:
        return 0.0
    
    cumsum_diff = 0.0
    result = 0.0
    for i in range(n - 1):
        cumsum_diff += u[i] - v[i]
        if cumsum_diff >= 0.0:
            result += cumsum_diff
        else:
            result -= cumsum_diff
    return result

class Solver:
    def __init__(self):
        # Trigger JIT compilation during init (excluded from runtime)
        dummy = np.array([1.0, 0.0], dtype=np.float64)
        wasserstein_1d(dummy, dummy)
    
    def solve(self, problem, **kwargs):
        u = np.ascontiguousarray(problem["u"], dtype=np.float64)
        v = np.ascontiguousarray(problem["v"], dtype=np.float64)
        return wasserstein_1d(u, v)