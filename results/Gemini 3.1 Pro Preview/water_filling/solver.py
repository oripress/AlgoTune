import numpy as np
from typing import Any
from numba import njit
@njit(cache=True, fastmath=True)
def _solve_numba(alpha, P_total):
    n = len(alpha)
    
    for i in range(n):
        if alpha[i] <= 0:
            return np.empty(0, dtype=np.float64), np.nan
            
    a_sorted = np.sort(alpha)
    
    prefix = 0.0
    w = 0.0
    for k in range(1, n + 1):
        prefix += a_sorted[k - 1]
        w_candidate = (P_total + prefix) / k
        if k == n or w_candidate <= a_sorted[k]:
            w = w_candidate
            break
            
    x_opt = np.empty(n, dtype=np.float64)
    sum_x = 0.0
    for i in range(n):
        val = w - alpha[i]
        if val > 0:
            x_opt[i] = val
            sum_x += val
        else:
            x_opt[i] = 0.0
            
    if sum_x > 0:
        scaling = P_total / sum_x
        for i in range(n):
            x_opt[i] *= scaling
            
    capacity = 0.0
    for i in range(n):
        capacity += np.log(alpha[i] + x_opt[i])
        
    return x_opt, capacity

class Solver:
    def __init__(self):
        # Trigger Numba compilation during initialization
        _solve_numba(np.array([1.0, 2.0], dtype=np.float64), 1.0)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        alpha = problem["alpha"]
        if type(alpha) is list:
            alpha_arr = np.fromiter(alpha, dtype=np.float64, count=len(alpha))
        else:
            alpha_arr = np.asarray(alpha, dtype=np.float64)
            
        P_total = float(problem["P_total"])
        n = len(alpha_arr)
        
        if n == 0 or P_total <= 0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
            
        x_opt, capacity = _solve_numba(alpha_arr, P_total)
        
        if len(x_opt) == 0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
            
        return {"x": x_opt.tolist(), "Capacity": float(capacity)}