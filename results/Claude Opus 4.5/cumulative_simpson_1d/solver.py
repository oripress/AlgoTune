import numpy as np
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def cumulative_simpson_numba(y, dx):
    n = len(y)
    
    if n < 2:
        return np.empty(0, dtype=np.float64)
    
    result = np.empty(n - 1, dtype=np.float64)
    h = dx
    
    if n == 2:
        result[0] = h * (y[0] + y[1]) / 2.0
        return result
    
    h3 = h / 3.0
    h12 = h / 12.0
    
    # First two elements
    result[0] = h12 * (5.0*y[0] + 8.0*y[1] - y[2])
    result[1] = h3 * (y[0] + 4.0*y[1] + y[2])
    
    if n == 3:
        return result
    
    # Compute odd indices (3, 5, 7, ...)
    # Number of odd indices from 3 to n-2
    max_odd = n - 2 if (n - 2) % 2 == 1 else n - 3
    num_odd = (max_odd - 3) // 2 + 1 if max_odd >= 3 else 0
    
    if num_odd > 0:
        # Compute all increments for odd indices
        cumsum_val = 0.0
        for i in range(num_odd):
            k = 3 + 2 * i
            cumsum_val += h3 * (y[k-1] + 4.0*y[k] + y[k+1])
            result[k] = result[1] + cumsum_val
    
    # Compute even indices (2, 4, 6, ...)
    max_even = n - 2 if (n - 2) % 2 == 0 else n - 3
    num_even = (max_even - 2) // 2 + 1 if max_even >= 2 else 0
    
    for i in range(num_even):
        k = 2 + 2 * i
        result[k] = result[k-1] + h12 * (-y[k-2] + 8.0*y[k-1] + 5.0*y[k])
    
    return result

class Solver:
    def solve(self, problem, **kwargs):
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        return cumulative_simpson_numba(y, dx)