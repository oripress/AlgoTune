import numpy as np
from numpy.typing import NDArray
import numba as nb

@nb.njit(cache=True, fastmath=True)
def cumulative_simpson_fast(y, dx):
    n = len(y)
    
    # scipy's cumulative_simpson returns n-1 values for n input points
    if n < 3:
        # For less than 3 points, return empty or single trapezoid
        if n <= 1:
            return np.empty(0, dtype=np.float64)
        else:
            return np.array([0.5 * dx * (y[0] + y[1])], dtype=np.float64)
    
    # For n points, we have n-1 intervals
    result = np.zeros(n-1, dtype=np.float64)
    
    # First interval - trapezoidal rule
    result[0] = 0.5 * dx * (y[0] + y[1])
    
    # Apply Simpson's rule for remaining intervals
    for i in range(2, n):
        idx = i - 1  # result index
        if i % 2 == 0:
            # Even index in y array - can use Simpson's rule
            result[idx] = result[idx-2] + dx / 3.0 * (y[i-2] + 4.0*y[i-1] + y[i])
        else:
            # Odd index - use trapezoidal for last interval
            result[idx] = result[idx-1] + 0.5 * dx * (y[i-1] + y[i])
    
    return result

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        
        # Use numba-compiled function
        result = cumulative_simpson_fast(y, dx)
        
        return result