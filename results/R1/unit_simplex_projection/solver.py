import numpy as np
from numba import jit

@jit(nopython=True)
def _simplex_projection(y):
    n = len(y)
    if n == 0:
        return np.zeros(0)
    
    # Initialize variables
    cumsum = 0.0
    k = 0
    max_val = -np.inf
    max_idx = -1
    
    # First pass to find initial maximum
    for i in range(n):
        if y[i] > max_val:
            max_val = y[i]
            max_idx = i
    
    # If no valid element found, return zeros
    if max_idx == -1:
        return np.zeros(n)
    
    # Initialize active array and mark first max as inactive
    active = np.ones(n, dtype=np.bool_)
    active[max_idx] = False
    cumsum += max_val
    k = 1
    
    # Iterate to find additional elements
    for _ in range(1, n):
        max_val = -np.inf
        max_idx = -1
        
        # Find next maximum among active elements
        for i in range(n):
            if active[i] and y[i] > max_val:
                max_val = y[i]
                max_idx = i
                
        if max_idx == -1:
            break
            
        # Check condition: k * max_val > cumsum - 1
        if k * max_val > cumsum - 1:
            cumsum += max_val
            k += 1
            active[max_idx] = False
        else:
            break
            
    # Compute optimal theta
    theta = (cumsum - 1) / k if k > 0 else 0
    
    # Compute projection
    return np.maximum(y - theta, 0)

class Solver:
    def solve(self, problem, **kwargs):
        """Optimized Euclidean projection using partial sort and early termination."""
        y_arr = np.array(problem["y"], dtype=np.float64)
        x = _simplex_projection(y_arr)
        return {"solution": x}