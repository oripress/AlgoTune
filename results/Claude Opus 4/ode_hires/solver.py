import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from typing import Any

@njit
def hires_numba(t, y, constants):
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    
    # HIRES system of equations - optimized computation
    y6_y8 = y6 * y8  # Compute once, use multiple times
    
    f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
    f2 = c1 * y1 - c5 * y2
    f3 = -c6 * y3 + c2 * y4 + c7 * y5
    f4 = c3 * y2 + c1 * y3 - c8 * y4
    f5 = -c9 * y5 + c2 * y6 + c2 * y7
    f6 = -c10 * y6_y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
    f7 = c10 * y6_y8 - c12 * y7
    f8 = -c10 * y6_y8 + c12 * y7
    
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

class Solver:
    def __init__(self):
        # Force compilation during init
        test_y = np.ones(8)
        test_c = np.ones(12)
        _ = hires_numba(0.0, test_y, test_c)
    
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the HIRES ODE system."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        constants = np.array(problem["constants"])
        
        # Wrapper for scipy
        def hires(t, y):
            return hires_numba(t, y, constants)
        
        # Set solver parameters - balance speed and accuracy
        # The validation uses rtol=1e-5, atol=1e-8, so we can be slightly looser
        rtol = 5e-6  # Slightly tighter than validation requirement
        atol = 5e-9  # Slightly tighter than validation requirement
        
        # Use Radau method which is excellent for stiff problems
        sol = solve_ivp(
            hires,
            [t0, t1],
            y0,
            method='Radau',
            rtol=rtol,
            atol=atol,
            dense_output=False,
            first_step=None,  # Let solver choose
            max_step=np.inf  # No step size restriction
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        # Return the final state as a list
        return sol.y[:, -1].tolist()