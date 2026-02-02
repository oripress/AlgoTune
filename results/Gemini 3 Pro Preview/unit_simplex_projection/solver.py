import numpy as np
from numba import njit
from typing import Any

@njit(cache=True, fastmath=True)
def find_theta_asc(u):
    n = len(u)
    current_sum = 0.0
    num_active = 0
    sum_active = 0.0
    
    # Iterate from the end (largest elements)
    for i in range(n):
        idx = n - 1 - i
        val = u[idx]
        current_sum += val
        count = i + 1
        
        # Condition: val > (current_sum - 1) / count
        if val * count > current_sum - 1.0:
            num_active = count
            sum_active = current_sum
        else:
            break
            
    theta = (sum_active - 1.0) / num_active
    return theta

class Solver:
    def __init__(self):
        # Warmup numba function
        dummy = np.array([0.2, 0.5, 1.0], dtype=np.float64)
        find_theta_asc(dummy)

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y_in = problem.get("y")
        
        # Use asarray to avoid copy if possible
        y = np.asarray(y_in, dtype=np.float64)
        
        if y.ndim > 1:
            y = y.ravel()
            
        # We need a copy for sorting anyway because we need original y for projection
        # np.sort returns a copy
        u = np.sort(y)
        
        theta = find_theta_asc(u)
        
        # Compute x
        x = np.subtract(y, theta)
        np.maximum(x, 0, out=x)
        
        return {"solution": x}