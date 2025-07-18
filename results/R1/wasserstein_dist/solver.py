import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy')
def compute_wasserstein(u_arr, v_arr):
    n = len(u_arr)
    total_u = 0.0
    total_v = 0.0
    cum_diff = 0.0
    total_cost = 0.0
    
    for i in range(n):
        # Update totals
        total_u += u_arr[i]
        total_v += v_arr[i]
        
        # Update cumulative difference
        cum_diff += u_arr[i] - v_arr[i]
        total_cost += abs(cum_diff)
    
    # Handle zero/negative totals
    if total_u <= 0 or total_v <= 0:
        if total_u == 0 and total_v == 0:
            return 0.0
        return float(n)
    
    # Apply normalization
    return total_cost / (total_u * total_v) * min(total_u, total_v)

class Solver:
    def solve(self, problem, **kwargs):
        u = problem['u']
        v = problem['v']
        n_val = len(u)
        if len(v) != n_val:
            return float(n_val)
        
        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        
        return float(compute_wasserstein(u_arr, v_arr))