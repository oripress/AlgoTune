import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        """
        y = np.asarray(problem["y"])
        dx = problem["dx"]
        n = len(y)
        
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([])
        if n == 2:
            return np.array([0.5 * dx * (y[0] + y[1])])
        
        # cumulative_simpson returns n-1 points (excluding the first point which is always 0)
        result = np.zeros(n - 1)
        
        # For Simpson's rule, we need at least 3 points
        # Handle first point with trapezoidal rule
        result[0] = 0.5 * dx * (y[0] + y[1])
        
        # For the rest, use extended Simpson's rule
        # Precompute cumulative sums for efficiency
        
        # Build arrays for odd and even indexed values
        odd_cumsum = np.zeros(n)
        even_cumsum = np.zeros(n)
        
        # Cumulative sums of odd-indexed elements (1, 3, 5, ...)
        odd_vals = y[1::2]
        odd_cumsum[1::2] = np.cumsum(odd_vals)
        # Propagate to even indices
        odd_cumsum[2::2] = odd_cumsum[1::2][:len(odd_cumsum[2::2])]
        
        # Cumulative sums of even-indexed elements (2, 4, 6, ...)
        if n > 2:
            even_vals = y[2::2]
            even_cumsum[2::2] = np.cumsum(even_vals)
            # Propagate to odd indices
            if n > 3:
                even_cumsum[3::2] = even_cumsum[2::2][:len(even_cumsum[3::2])]
        
        # Apply Simpson's rule
        for i in range(1, n - 1):
            idx = i + 1  # Index in original array
            if idx % 2 == 0:
                # Even index - use composite Simpson's
                s = y[0] + y[idx] + 4 * odd_cumsum[idx] + 2 * even_cumsum[idx]
                result[i] = dx * s / 3
            else:
                # Odd index - use previous Simpson's value plus correction
                result[i] = result[i-1] + dx / 24 * (9*y[idx] + 19*y[idx-1] - 5*y[idx-2] + y[idx-3])
        
        return result