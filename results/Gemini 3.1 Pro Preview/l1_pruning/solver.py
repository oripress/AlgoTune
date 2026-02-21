import numpy as np
from numba import njit
from typing import Any

@njit(cache=True)
def solve_numba(v, k):
    n = len(v)
    u = np.empty(n, dtype=np.float64)
    for i in range(n):
        u[i] = abs(v[i])
        
    # Linear time projection onto L1 ball
    # We need to find theta such that sum(max(u - theta, 0)) <= k
    # If sum(u) <= k, theta = 0
    sum_u = 0.0
    for i in range(n):
        sum_u += u[i]
        
    if sum_u <= k:
        theta = 0.0
    else:
        # We can use a randomized pivot algorithm or just sort for simplicity
        # Let's stick to sort for now but optimize the loop
        mu = np.sort(u)
        cumsum = 0.0
        theta = 0.0
        for j in range(n):
            mu_j = mu[n - 1 - j]
            cumsum += mu_j
            if mu_j < (cumsum - k) / (j + 1):
                theta = (cumsum - k) / (j + 1)
                break

    pruned = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = u[i] - theta
        if val > 0:
            if v[i] > 0:
                pruned[i] = val
            elif v[i] < 0:
                pruned[i] = -val
                
    return pruned

class Solver:
    def __init__(self):
        # Pre-compile for float64 and int64 arrays
        solve_numba(np.array([1.0, -2.0], dtype=np.float64), 1.0)
        solve_numba(np.array([1, -2], dtype=np.int64), 1.0)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        v = np.array(problem.get("v"), copy=False).reshape(-1)
        k = float(problem.get("k"))
        pruned = solve_numba(v, k)
        return {"solution": pruned}