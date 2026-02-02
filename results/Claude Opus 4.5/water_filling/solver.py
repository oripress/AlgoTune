from typing import Any
import numpy as np
import numba

@numba.jit(nopython=True, cache=True, fastmath=True, parallel=False)
def water_filling_core(alpha, P_total):
    n = len(alpha)
    idx = np.argsort(alpha)
    
    # Build sorted array and prefix sums
    a_sorted = np.empty(n, dtype=np.float64)
    prefix = np.empty(n, dtype=np.float64)
    a_sorted[0] = alpha[idx[0]]
    prefix[0] = a_sorted[0]
    for i in range(1, n):
        a_sorted[i] = alpha[idx[i]]
        prefix[i] = prefix[i-1] + a_sorted[i]
    
    # Find water level w
    w = (P_total + prefix[n-1]) / n
    for k in range(1, n + 1):
        w_candidate = (P_total + prefix[k - 1]) / k
        if k == n or w_candidate <= a_sorted[k]:
            w = w_candidate
            break
    
    # Compute x_opt
    x_opt = np.empty(n, dtype=np.float64)
    x_sum = 0.0
    for i in range(n):
        val = w - alpha[i]
        if val > 0.0:
            x_opt[i] = val
            x_sum += val
        else:
            x_opt[i] = 0.0
    
    # Rescale and compute capacity
    capacity = 0.0
    if x_sum > 0.0:
        scaling = P_total / x_sum
        for i in range(n):
            x_opt[i] *= scaling
            capacity += np.log(alpha[i] + x_opt[i])
    else:
        for i in range(n):
            capacity += np.log(alpha[i])
    
    return x_opt, capacity

@numba.jit(nopython=True, cache=True)
def check_all_positive(alpha):
    for i in range(len(alpha)):
        if alpha[i] <= 0.0:
            return False
    return True

class Solver:
    def __init__(self):
        # Warm up Numba JIT
        test_alpha = np.array([1.0, 2.0, 3.0])
        water_filling_core(test_alpha, 1.0)
        check_all_positive(test_alpha)
    
    def solve(self, problem, **kwargs) -> Any:
        alpha_list = problem["alpha"]
        P_total = float(problem["P_total"])
        n = len(alpha_list)
        
        if n == 0 or P_total <= 0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        # Fast conversion - use frombuffer when possible
        if isinstance(alpha_list, np.ndarray):
            alpha = alpha_list.astype(np.float64, copy=False)
        else:
            alpha = np.fromiter(alpha_list, dtype=np.float64, count=n)
        
        if not check_all_positive(alpha):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        x_opt, capacity = water_filling_core(alpha, P_total)
        
        # Directly return numpy array's tolist
        return {"x": x_opt.tolist(), "Capacity": capacity}