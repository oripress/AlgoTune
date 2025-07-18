import numpy as np
from typing import Any
import numba

# Use Numba's JIT compiler. We remove `fastmath=True` as it can
# introduce small numerical errors that fail the tolerance check.
# Correctness is more important than the minor speedup from fastmath.
@numba.njit(cache=True)
def _cumulative_simpson_numba(y, dx):
    """
    Numba-jitted implementation of cumulative Simpson's rule.
    The input 3D array is reshaped to 2D to allow for a single, simple loop.
    """
    d1, d2, N = y.shape
    if N < 2:
        return np.empty((d1, d2, 0), dtype=np.float64)

    # Reshape for processing over a single dimension
    y_reshaped = y.reshape(d1 * d2, N)
    res_reshaped = np.empty((d1 * d2, N - 1), dtype=np.float64)

    # Loop over all 1D slices using standard `range`.
    for i in range(d1 * d2):
        y_i = y_reshaped[i]
        res_i = res_reshaped[i]

        # 1. Calculate integrals to even points (y[2], y[4], ...).
        if N > 2:
            simpson_sum = 0.0
            for k in range(1, (N - 1) // 2 + 1):
                idx = 2 * k
                term = (y_i[idx - 2] + 4.0 * y_i[idx - 1] + y_i[idx]) * (dx / 3.0)
                simpson_sum += term
                res_i[idx - 1] = simpson_sum

        # 2. Calculate integrals to odd points (y[1], y[3], ...).
        if N > 1:
            res_i[0] = (y_i[0] + y_i[1]) * (dx / 2.0)
            if N > 3:
                for k in range(1, (N - 2) // 2 + 1):
                    idx = 2 * k
                    base_integral = res_i[idx - 1]
                    trapezoid_corr = (y_i[idx] + y_i[idx + 1]) * (dx / 2.0)
                    res_i[idx] = base_integral + trapezoid_corr
    
    # Reshape result back, passing the shape as a tuple.
    return res_reshaped.reshape((d1, d2, N - 1))

class Solver:
    """
    A solver that uses a Numba-jitted, serial implementation of
    cumulative_simpson for high performance.
    """
    def __init__(self):
        """
        Numba JIT compilation happens on the first call.
        """
        pass

    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the cumulative integral using the pre-compiled Numba function.
        """
        y2 = problem["y2"]
        dx = problem["dx"]
        
        # Ensure input is float64 for consistency with reference
        y2_64 = y2.astype(np.float64)

        return _cumulative_simpson_numba(y2_64, float(dx))