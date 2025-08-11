import numpy as np
from numpy.typing import NDArray
from numba import jit

@jit(nopython=True)
def _cumulative_simpson_numba_max(y, dx, result):
    """Maximum optimized Numba-compiled function for cumulative Simpson integration."""
    n = len(y)
    
    # Handle edge cases
    if n <= 1:
        return
    
    # First element uses trapezoidal rule
    result[0] = 0.5 * dx * (y[0] + y[1])
    
    if n == 2:
        return
    
    # Pre-compute constants
    dx_third = dx / 3.0
    dx_half = 0.5 * dx
    four_dx_third = 4.0 * dx_third
    
    # Optimized processing with minimal branching and pre-computed constants
    i = 0
    simpson_sum = 0.0
    
    # Process in chunks of 12 points (6 Simpson segments) for maximum cache locality
    while i + 11 < n:
        # First Simpson segment (points i, i+1, i+2)
        simpson_sum += dx_third * y[i] + four_dx_third * y[i+1] + dx_third * y[i+2]
        result[i+1] = simpson_sum
        
        # Trapezoidal segment (points i+1, i+2)
        result[i+2] = result[i+1] + dx_half * (y[i+1] + y[i+2])
        
        # Second Simpson segment (points i+2, i+3, i+4)
        simpson_sum += dx_third * y[i+2] + four_dx_third * y[i+3] + dx_third * y[i+4]
        result[i+3] = simpson_sum
        
        # Trapezoidal segment (points i+3, i+4)
        result[i+4] = result[i+3] + dx_half * (y[i+3] + y[i+4])
        
        # Third Simpson segment (points i+4, i+5, i+6)
        simpson_sum += dx_third * y[i+4] + four_dx_third * y[i+5] + dx_third * y[i+6]
        result[i+5] = simpson_sum
        
        # Trapezoidal segment (points i+5, i+6)
        result[i+6] = result[i+5] + dx_half * (y[i+5] + y[i+6])
        
        # Fourth Simpson segment (points i+6, i+7, i+8)
        simpson_sum += dx_third * y[i+6] + four_dx_third * y[i+7] + dx_third * y[i+8]
        result[i+7] = simpson_sum
        
        # Trapezoidal segment (points i+7, i+8)
        result[i+8] = result[i+7] + dx_half * (y[i+7] + y[i+8])
        
        # Fifth Simpson segment (points i+8, i+9, i+10)
        simpson_sum += dx_third * y[i+8] + four_dx_third * y[i+9] + dx_third * y[i+10]
        result[i+9] = simpson_sum
        
        # Trapezoidal segment (points i+9, i+10)
        result[i+10] = result[i+9] + dx_half * (y[i+9] + y[i+10])
        
        # Sixth Simpson segment (points i+10, i+11, i+12)
        simpson_sum += dx_third * y[i+10] + four_dx_third * y[i+11] + dx_third * y[i+12]
        result[i+11] = simpson_sum
        
        # Trapezoidal segment (points i+11, i+12)
        result[i+12] = result[i+11] + dx_half * (y[i+11] + y[i+12])
        
        i += 12
    
    # Handle remaining points with minimal branching
    while i + 2 < n:
        # Simpson segment with optimized calculation
        simpson_sum += dx_third * y[i] + four_dx_third * y[i+1] + dx_third * y[i+2]
        result[i+1] = simpson_sum
        
        if i + 3 < n:
            # Trapezoidal segment
            result[i+2] = result[i+1] + dx_half * (y[i+1] + y[i+2])
            i += 2
        else:
            break

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        y = np.array(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        n = len(y)
        
        if n <= 1:
            return np.array([], dtype=np.float64)
        
        # Initialize result array
        result = np.zeros(n - 1, dtype=np.float64)
        
        # Use maximum optimized Numba-compiled function
        _cumulative_simpson_numba_max(y, dx, result)
        
        return result