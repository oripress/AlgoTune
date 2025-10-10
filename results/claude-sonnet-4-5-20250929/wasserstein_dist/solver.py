import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def compute_wasserstein(u, v):
    """Compute Wasserstein distance using JIT-compiled code."""
    n = len(u)
    if n <= 1:
        return 0.0
    
    cum_u = 0.0
    cum_v = 0.0
    distance = 0.0
    
    for i in range(n - 1):
        cum_u += u[i]
        cum_v += v[i]
        diff = cum_u - cum_v
        distance += abs(diff)
    
    return distance

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute Wasserstein distance for discrete distributions on [1,2,...,n]
        using JIT-compiled code for maximum performance.
        """
        u = np.asarray(problem["u"], dtype=np.float64)
        v = np.asarray(problem["v"], dtype=np.float64)
        
        return compute_wasserstein(u, v)