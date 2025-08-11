import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem: dict, **kwargs) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        y = problem["y"]
        dx = problem["dx"]
        
        # Use scipy's cumulative_simpson without initial to match reference behavior
        result = cumulative_simpson(y, dx=dx)
        return result
        # For Simpson's rule, we need at least 3 points
        if n < 3:
            # Fall back to trapezoidal rule for small arrays
            for i in range(1, n):
                result[i] = result[i-1] + 0.5 * dx * (y[i-1] + y[i])
            return result
        
        # For the first interval, use trapezoidal rule
        result[1] = 0.5 * dx * (y[0] + y[1])
        
        # For subsequent points, use Simpson's rule where possible
        for i in range(2, n):
            # Use Simpson's rule for every pair of intervals
            if i % 2 == 0:
                # Even index: we can apply Simpson's rule
                result[i] = result[i-2] + dx/3.0 * (y[i-2] + 4*y[i-1] + y[i])
            else:
                # Odd index: use trapezoidal rule to maintain continuity
                result[i] = result[i-1] + 0.5 * dx * (y[i-1] + y[i])
                
        return result