import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True)
def cumulative_simpson_1d(y, dx):
    n = y.size
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    res = np.zeros(n-1, dtype=np.float64)
    
    # First interval: trapezoidal rule
    res[0] = (y[0] + y[1]) * dx / 2.0
    
    if n < 3:
        return res
        
    # Second interval: Simpson's rule
    res[1] = (y[0] + 4*y[1] + y[2]) * dx / 3.0
    
    # Process remaining points
    for k in range(3, n):
        if k % 2 == 1:  # Odd index (trapezoidal)
            res[k-1] = res[k-2] + (y[k-1] + y[k]) * dx / 2.0
        else:  # Even index (Simpson)
            res[k-1] = res[k-3] + (y[k-2] + 4*y[k-1] + y[k]) * dx / 3.0
            
    return res

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        y2 = problem["y2"]
        dx = problem["dx"]
        n0, n1, n2 = y2.shape
        
        # Create output array with shape (n0, n1, n2-1)
        result = np.zeros((n0, n1, n2-1), dtype=np.float64)
        
        # Process each 1D signal
        for i in range(n0):
            for j in range(n1):
                result[i, j] = cumulative_simpson_1d(y2[i, j], dx)
        
        return result