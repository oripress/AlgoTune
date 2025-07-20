import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True, boundscheck=False, parallel=True)
def cumulative_simpson_fast(y, dx):
    n = len(y)
    if n < 2:
        return np.empty(0, dtype=np.float64)
    
    res = np.zeros(n-1, dtype=np.float64)
    c1 = dx / 3.0
    c2 = dx * 0.5
    
    # Handle small arrays
    res[0] = c2 * (y[0] + y[1])
    if n < 3:
        return res
        
    res[1] = c1 * (y[0] + 4.0 * y[1] + y[2])
    if n < 4:
        return res
        
    # Precompute initial values
    last_simpson = res[1]
    last_trap = res[0]
    
    # Main processing loop - unrolled and specialized
    i = 3
    while i < n - 2:
        # Process two intervals per iteration
        # First: Simpson step for [i-1, i+1]
        new_simpson = last_simpson + c1 * (y[i-1] + 4.0 * y[i] + y[i+1])
        res[i] = new_simpson
        
        # Second: Trapezoidal step for [i+1, i+2]
        new_trap = new_simpson + c2 * (y[i+1] + y[i+2])
        res[i+1] = new_trap
        
        # Update state for next iteration
        last_simpson = new_simpson
        last_trap = new_trap
        i += 2
    
    # Handle remaining points
    if i < n:
        res[i-1] = last_simpson + c2 * (y[i-1] + y[i])

    return res

class Solver:
    def solve(self, problem, **kwargs):
        y = np.ascontiguousarray(problem['y'], dtype=np.float64)
        dx = problem['dx']
        return cumulative_simpson_fast(y, dx)