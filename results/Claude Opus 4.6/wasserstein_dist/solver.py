import numpy as np
from numba import njit

@njit(cache=True)
def _wasserstein(u, v):
    n = len(u)
    u_sum = 0.0
    v_sum = 0.0
    for i in range(n):
        u_sum += u[i]
        v_sum += v[i]
    
    cdf_diff = 0.0
    result = 0.0
    inv_u = 1.0 / u_sum
    inv_v = 1.0 / v_sum
    for i in range(n):
        cdf_diff += u[i] * inv_u - v[i] * inv_v
        if cdf_diff > 0:
            result += cdf_diff
        else:
            result -= cdf_diff
    return result

try:
    from wasserstein_cy import wasserstein_from_list
    _has_cy = True
except ImportError:
    _has_cy = False

class Solver:
    def __init__(self):
        # Warm up numba
        _wasserstein(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    
    def solve(self, problem, **kwargs):
        u = problem["u"]
        v = problem["v"]
        if _has_cy and isinstance(u, list):
            return wasserstein_from_list(u, v)
        if not isinstance(u, np.ndarray):
            u = np.asarray(u, dtype=np.float64)
        elif u.dtype != np.float64:
            u = u.astype(np.float64)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float64)
        elif v.dtype != np.float64:
            v = v.astype(np.float64)
        return float(_wasserstein(u, v))