import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True)
def cumulative_simpson_numba(y, dx):
    n = len(y)
    if n < 2:
        return np.zeros(0)
    n_intervals = n - 1
    res = np.zeros(n_intervals)
    
    # Precompute constants
    dx_half = dx * 0.5
    dx_third = dx / 3.0
    
    # First interval (trapezoidal rule)
    res[0] = (y[0] + y[1]) * dx_half
    if n_intervals == 1:
        return res
    
    # Second interval (Simpson's rule)
    res[1] = (y[0] + 4*y[1] + y[2]) * dx_third
    if n_intervals == 2:
        return res
    
    # Number of full blocks (2 intervals per block)
    num_blocks = (n_intervals - 2) // 2
    leftover = (n_intervals - 2) % 2
    
    # Process full blocks
    for i in range(num_blocks):
        j = 2 + 2*i  # Start index for current block
        prev_idx = 1 + 2*i  # Previous result index
        res_idx1 = 2 + 2*i   # Result index for first interval
        res_idx2 = 3 + 2*i   # Result index for second interval
        
        # Trapezoid for first interval in block
        res[res_idx1] = res[prev_idx] + (y[j] + y[j+1]) * dx_half
        # Simpson for second interval in block
        res[res_idx2] = res[prev_idx] + (y[j] + 4*y[j+1] + y[j+2]) * dx_third
    
    # Handle leftover interval if exists
    if leftover:
        j = 2 + 2*num_blocks
        res_idx = 2 + 2*num_blocks
        res[res_idx] = res[2*num_blocks+1] + (y[j] + y[j+1]) * dx_half
    
    return res

class Solver:
    def solve(self, problem, **kwargs):
        y = problem['y']
        dx = problem['dx']
        y_arr = np.asarray(y)
        return cumulative_simpson_numba(y_arr, dx)