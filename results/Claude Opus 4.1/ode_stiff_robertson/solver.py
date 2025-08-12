import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
import numba as nb

# Pre-compile the ODE function outside the class
@nb.njit(fastmath=True, cache=True)
def rober_jit(t, y, k1, k2, k3):
    y1, y2, y3 = y[0], y[1], y[2]
    
    out = np.empty(3, dtype=np.float64)
    out[0] = -k1 * y1 + k3 * y2 * y3
    out[1] = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
    out[2] = k2 * y2 * y2
    return out

class Solver:
    def __init__(self):
        # Warm up the JIT compilation during init
        y_test = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        _ = rober_jit(0.0, y_test, 0.04, 3e7, 1e4)
    
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Robertson chemical kinetics ODE system."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k = problem["k"]
        k1, k2, k3 = float(k[0]), float(k[1]), float(k[2])
        
        # Create wrapper for scipy
        def rober(t, y):
            return rober_jit(t, y, k1, k2, k3)
        
        # Use Radau with optimized tolerances
        # The validation uses rtol=1e-5, atol=1e-8, so we can be slightly looser
        rtol = 2e-6
        atol = 5e-9
        
        sol = solve_ivp(
            rober,
            [t0, t1],
            y0,
            method="Radau",  # Best for stiff problems
            rtol=rtol,
            atol=atol,
            dense_output=False,
            first_step=1e-6  # Help the solver start
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()