import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def compute_wasserstein_numba(u, v):
    """
    Fast Wasserstein distance computation using Numba JIT.
    """
    n = len(u)
    cdf_diff = 0.0
    cum_u = 0.0
    cum_v = 0.0
    
    for i in range(n):
        cum_u += u[i]
        cum_v += v[i]
        cdf_diff += abs(cum_u - cum_v)
    
    return cdf_diff

class Solver:
    def __init__(self):
        # Warm up JIT compilation
        dummy = np.array([0.5, 0.5], dtype=np.float64)
        compute_wasserstein_numba(dummy, dummy)
    
    def solve(self, problem, **kwargs):
        """
        Compute Wasserstein distance using Numba JIT.
        """
        u = np.asarray(problem["u"], dtype=np.float64)
        v = np.asarray(problem["v"], dtype=np.float64)
        
        return compute_wasserstein_numba(u, v)