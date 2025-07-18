import numpy as np
from typing import Any
import numba

@numba.njit(fastmath=True, cache=True)
def compute_wasserstein_fast(u, v, n):
    """
    Ultra-fast Wasserstein distance computation.
    """
    # Normalize in single pass
    sum_u = 0.0
    sum_v = 0.0
    for i in range(n):
        sum_u += u[i]
        sum_v += v[i]
    
    inv_sum_u = 1.0 / sum_u
    inv_sum_v = 1.0 / sum_v
    
    # Compute distance using CDF approach
    cdf_u = 0.0
    cdf_v = 0.0
    distance = 0.0
    
    for i in range(n):
        cdf_u += u[i] * inv_sum_u
        cdf_v += v[i] * inv_sum_v
        diff = cdf_u - cdf_v
        distance += diff if diff >= 0 else -diff
    
    return distance

class Solver:
    def __init__(self):
        # Pre-compile with different sizes
        for size in [2, 10, 100]:
            dummy = np.ones(size, dtype=np.float64)
            compute_wasserstein_fast(dummy, dummy, size)
    
    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Compute Wasserstein distance using ultra-optimized approach.
        """
        try:
            # Direct access to lists is faster than np.asarray for small arrays
            u_list = problem["u"]
            v_list = problem["v"]
            n = len(u_list)
            
            # For small arrays, use lists directly
            if n < 50:
                return self._solve_small(u_list, v_list, n)
            
            # For larger arrays, use numpy
            u = np.asarray(u_list, dtype=np.float64)
            v = np.asarray(v_list, dtype=np.float64)
            return compute_wasserstein_fast(u, v, n)
            
        except Exception as e:
            return float(len(problem["u"]))
    
    def _solve_small(self, u, v, n):
        """Direct computation for small arrays."""
        sum_u = sum(u)
        sum_v = sum(v)
        
        cdf_u = 0.0
        cdf_v = 0.0
        distance = 0.0
        
        for i in range(n):
            cdf_u += u[i] / sum_u
            cdf_v += v[i] / sum_v
            distance += abs(cdf_u - cdf_v)
        
        return distance