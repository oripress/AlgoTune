import numpy as np
from numpy.typing import NDArray
from numba import njit

@njit(fastmath=True, cache=True)
def _cumulative_simpson_numba(y: NDArray, dx: float) -> NDArray:
    n = y.shape[0]
    res = np.empty(n, dtype=np.float64)
    res[0] = 0.0
    
    if n >= 3:
        dx_3 = dx / 3.0
        dx_12 = dx / 12.0
        
        # Main loop for Simpson's rule
        # We process in steps of 2
        # res[i] is the integral from 0 to i
        
        current_sum = 0.0
        
        # Loop over even indices 2, 4, ..., up to N-1 or N-2
        # We stop at n-1 because we access y[i], y[i-1], y[i-2]
        # The loop range is for the 'end' of the interval
        
        # We can iterate i from 2 to n-1 with step 2
        # If n is odd (e.g. 5), indices are 0, 1, 2, 3, 4. Loop: 2, 4.
        # If n is even (e.g. 4), indices are 0, 1, 2, 3. Loop: 2.
        
        limit = n - 1 if (n % 2 != 0) else n - 2
        
        i = 2
        # Unroll loop by 4 (process 4 Simpson intervals per iteration)
        while i <= limit - 6:
            # Interval 1: [i-2, i]
            y_im2 = y[i-2]
            y_im1 = y[i-1]
            y_i = y[i]
            
            res[i-1] = current_sum + dx_12 * (5.0 * y_im2 + 8.0 * y_im1 - y_i)
            current_sum += dx_3 * (y_im2 + 4.0 * y_im1 + y_i)
            res[i] = current_sum
            
            # Interval 2: [i, i+2]
            y_im2 = y_i
            y_im1 = y[i+1]
            y_i = y[i+2]
            
            res[i+1] = current_sum + dx_12 * (5.0 * y_im2 + 8.0 * y_im1 - y_i)
            current_sum += dx_3 * (y_im2 + 4.0 * y_im1 + y_i)
            res[i+2] = current_sum
            
            # Interval 3: [i+2, i+4]
            y_im2 = y_i
            y_im1 = y[i+3]
            y_i = y[i+4]
            
            res[i+3] = current_sum + dx_12 * (5.0 * y_im2 + 8.0 * y_im1 - y_i)
            current_sum += dx_3 * (y_im2 + 4.0 * y_im1 + y_i)
            res[i+4] = current_sum
            
            # Interval 4: [i+4, i+6]
            y_im2 = y_i
            y_im1 = y[i+5]
            y_i = y[i+6]
            
            res[i+5] = current_sum + dx_12 * (5.0 * y_im2 + 8.0 * y_im1 - y_i)
            current_sum += dx_3 * (y_im2 + 4.0 * y_im1 + y_i)
            res[i+6] = current_sum
            
            i += 8
            
        # Handle remaining iterations
        while i <= limit:
            y_im2 = y[i-2]
            y_im1 = y[i-1]
            y_i = y[i]
            
            res[i-1] = current_sum + dx_12 * (5.0 * y_im2 + 8.0 * y_im1 - y_i)
            current_sum += dx_3 * (y_im2 + 4.0 * y_im1 + y_i)
            res[i] = current_sum
            i += 2

    # Handle the last element if n is even
    if n % 2 == 0:
        if n >= 3:
            # Backward parabola for the last interval
            res[n-1] = res[n-2] + (dx / 12.0) * (-y[n-3] + 8.0 * y[n-2] + 5.0 * y[n-1])
        elif n == 2:
            # Trapezoidal rule fallback for 2 points
            res[1] = res[0] + (dx / 2.0) * (y[0] + y[1])
            
    return res[1:]

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        Optimized Numba implementation.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        return _cumulative_simpson_numba(y, dx)