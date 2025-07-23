import numpy as np
from typing import Any
from numba import njit

# Using fastmath=True can provide additional speedups by relaxing strict
# IEEE 754 compliance, which is effective for math-heavy loops.
@njit(fastmath=True)
def _numba_cumulative_simpson(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Numba-accelerated single-pass implementation of cumulative Simpson's rule.
    This version computes the cumulative sum directly in one loop, avoiding
    an intermediate array and a separate cumsum call for maximum efficiency.
    """
    N = y.shape[0]

    if N < 2:
        return np.empty(0, dtype=np.float64)

    # The result array has N-1 elements.
    res = np.empty(N - 1, dtype=np.float64)
    
    # Handle N=2 as a special case, as it's a simple trapezoid rule.
    if N == 2:
        res[0] = (dx / 2.0) * (y[0] + y[1])
        return res

    simpson_n = N
    is_even = (N % 2 == 0)
    if is_even:
        simpson_n = N - 1

    dx_div_12 = dx / 12.0
    current_integral = 0.0
    
    # Loop over pairs of intervals, applying the formulas and updating the cumulative sum.
    # This fuses the integration and summation steps into a single, cache-friendly loop.
    for i in range(0, simpson_n - 2, 2):
        y0, y1, y2 = y[i], y[i+1], y[i+2]
        
        # Integral over [x_i, x_{i+1}]
        integral1 = dx_div_12 * (5.0 * y0 + 8.0 * y1 - y2)
        current_integral += integral1
        res[i] = current_integral
        
        # Integral over [x_{i+1}, x_{i+2}]
        integral2 = dx_div_12 * (-y0 + 8.0 * y1 + 5.0 * y2)
        current_integral += integral2
        res[i+1] = current_integral

    # If N is even, the last interval is integrated using the trapezoidal rule.
    if is_even:
        trapezoid_integral = (dx / 2.0) * (y[N - 2] + y[N - 1])
        res[N-2] = current_integral + trapezoid_integral

    return res

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the cumulative integral of a one-dimensional function using Simpson’s rule.
        This version delegates the computation to a highly optimized, single-pass
        Numba-jitted helper function.
        """
        y = np.array(problem["y"], dtype=np.float64)
        dx = problem["dx"]
        
        return _numba_cumulative_simpson(y, dx)