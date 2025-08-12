import numpy as np
from typing import Any
import numba

@numba.njit(fastmath=True, cache=True)
def find_threshold(u, k):
    """Find the optimal threshold for L1 projection using numba JIT."""
    n = len(u)
    mu = np.sort(u)[::-1]
    
    cumsum = 0.0
    theta = 0.0
    for j in range(n):
        cumsum += mu[j]
        threshold = (cumsum - k) / (j + 1)
        if j == n - 1 or mu[j + 1] < threshold:
            theta = threshold
            break
    
    return theta

@numba.njit(fastmath=True, cache=True)
def project_l1_ball(v, k):
    """Project vector v onto L1 ball of radius k."""
    n = len(v)
    v_abs = np.abs(v)
    
    # Check if already in L1 ball
    if np.sum(v_abs) <= k:
        return v.copy()
    
    # Find optimal threshold
    theta = find_threshold(v_abs, k)
    
    # Apply soft thresholding
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        if v_abs[i] > theta:
            result[i] = (v_abs[i] - theta) * np.sign(v[i])
        else:
            result[i] = 0.0
    
    return result

class Solver:
    def __init__(self):
        # Warm up numba compilation
        dummy = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        _ = project_l1_ball(dummy, 1.0)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the L1 proximal operator problem using numba-accelerated code.
        """
        v = np.asarray(problem["v"], dtype=np.float64)
        k = float(problem["k"])
        
        solution = project_l1_ball(v, k)
        
        return {"solution": solution.tolist()}