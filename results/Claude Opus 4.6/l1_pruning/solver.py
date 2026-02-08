import numpy as np
from typing import Any
import numba

@numba.njit(cache=True)
def _partition_based_theta(u, k):
    """Find theta using partition-based O(n) expected time algorithm."""
    n = len(u)
    # We need to find the projection onto the simplex
    # Using the algorithm from Condat (2016) or similar
    
    # Make a working copy of abs values
    mu = u.copy()
    mu.sort()  # ascending
    
    cumsum = 0.0
    theta = 0.0
    for j in range(n):
        idx = n - 1 - j
        cumsum += mu[idx]
        thresh = (cumsum - k) / (j + 1)
        if mu[idx] < thresh:
            theta = thresh
            break
    
    return theta

@numba.njit(cache=True)
def _solve_l1_proj(v, k):
    n = len(v)
    u = np.empty(n)
    signs = np.empty(n)
    for i in range(n):
        val = v[i]
        if val >= 0:
            u[i] = val
            signs[i] = 1.0
        else:
            u[i] = -val
            signs[i] = -1.0
    
    theta = _partition_based_theta(u, k)
    
    for i in range(n):
        val = u[i] - theta
        if val > 0.0:
            v[i] = val * signs[i]
        else:
            v[i] = 0.0

# Warm up
_dummy_v = np.array([1.0, -1.0])
_solve_l1_proj(_dummy_v, 1.0)

def _solve_python(v_list, k):
    """Pure Python implementation for small arrays."""
    n = len(v_list)
    u = [abs(x) for x in v_list]
    mu = sorted(u, reverse=True)
    
    cumsum = 0.0
    theta = 0.0
    for j in range(n):
        cumsum += mu[j]
        thresh = (cumsum - k) / (j + 1)
        if mu[j] < thresh:
            theta = thresh
            break
    
    result = [0.0] * n
    for i in range(n):
        val = u[i] - theta
        if val > 0.0:
            if v_list[i] >= 0:
                result[i] = val
            else:
                result[i] = -val
    
    return result

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        v_list = problem["v"]
        k = float(problem["k"])
        n = len(v_list)
        
        if n < 200:
            return {"solution": _solve_python(v_list, k)}
        else:
            v = np.array(v_list, dtype=np.float64)
            _solve_l1_proj(v, k)
            return {"solution": v.tolist()}