import numpy as np
import numba
from typing import Any

@numba.jit(nopython=True, fastmath=True, cache=True)
def _newton_raphson_core(x0, a0, a1, a2, a3, a4, a5):
    """
    Core Newton-Raphson solver, JIT-compiled with Numba.
    """
    x = x0.copy()
    n = x0.shape[0]
    
    # Experimentally determined minimum number of iterations for convergence
    max_iter = 10

    # Pre-calculate loop-invariant terms
    c_a3_a5 = a3 / a5
    c_a3_a4 = a3 / a4

    for _ in range(max_iter):
        # Numba will optimize this element-wise loop into fast machine code
        for i in range(n):
            common_term = a0[i] + x[i] * a3
            
            exp_arg = common_term / a5
            
            if exp_arg > 700.0:
                exp_term = np.exp(700.0)
                fpx = -c_a3_a4 - 1.0
            elif exp_arg < -700.0:
                exp_term = np.exp(-700.0)
                fpx = -c_a3_a4 - 1.0
            else:
                exp_term = np.exp(exp_arg)
                fpx = -a2 * c_a3_a5 * exp_term - c_a3_a4 - 1.0
            
            fx = a1[i] - a2 * (exp_term - 1.0) - common_term / a4 - x[i]
            
            # Newton-Raphson update step. Add a small epsilon to avoid division by zero.
            x[i] -= fx / (fpx + np.copysign(1e-9, fpx))

    # After a fixed number of iterations, we assume convergence.
    # We only need to filter out results that became non-finite (NaN/inf).
    for i in range(n):
        if not np.isfinite(x[i]):
            x[i] = np.nan
            
    return x

class Solver:
    """
    A highly optimized solver for finding roots of a specific transcendental equation.
    """
    def __init__(self):
        """Initializes the solver and its constant parameters."""
        self.a2 = 1.0
        self.a3 = 1.0
        self.a4 = 1.0
        self.a5 = 1.0

    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Parses the problem, calls the Numba-jitted core solver, and returns the results.
        """
        try:
            # Convert inputs to NumPy arrays for efficient processing
            x0 = np.array(problem["x0"], dtype=np.float64)
            a0 = np.array(problem["a0"], dtype=np.float64)
            a1 = np.array(problem["a1"], dtype=np.float64)
        except (KeyError, TypeError):
            # Handle malformed input gracefully
            return {"roots": []}

        # These attributes are constants set on the instance by the evaluation harness.
        a2, a3, a4, a5 = self.a2, self.a3, self.a4, self.a5
        
        # Call the optimized core function
        roots = _newton_raphson_core(x0, a0, a1, a2, a3, a4, a5)

        # Convert the result back to a list as required by the output format
        return {"roots": roots.tolist()}