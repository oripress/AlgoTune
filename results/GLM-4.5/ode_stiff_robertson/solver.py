from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

# Numba-JIT compiled ODE function for maximum performance
@jit(nopython=True)
def rober_numba(t, y, k1, k2, k3):
    y1, y2, y3 = y
    # Precompute common terms
    k1_y1 = k1 * y1
    k2_y2_sq = k2 * y2 * y2
    k3_y2_y3 = k3 * y2 * y3
    f0 = -k1_y1 + k3_y2_y3
    f1 = k1_y1 - k2_y2_sq - k3_y2_y3
    f2 = k2_y2_sq
    return np.array([f0, f1, f2])

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Robertson chemical kinetics system with maximum speed optimizations."""
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k1, k2, k3 = problem["k"]
        
        # Use carefully balanced tolerances for speed and accuracy
        rtol = 1e-10  # More relaxed than reference but still accurate
        atol = 1e-8   # More relaxed than reference but still accurate

        # Try BDF method with optimized parameters - back to best settings
        sol = solve_ivp(
            lambda t, y: rober_numba(t, y, k1, k2, k3),
            [t0, t1],
            y0,
            method="BDF",
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
            first_step=1e-4,
            max_step=t1/2,
            # Additional optimization parameters
            jac_sparsity=None,  # No sparsity pattern for this small system
            vectorized=False,   # Not vectorized
        )
        
        if not sol.success:
            # Try LSODA as fallback
            sol = solve_ivp(
                lambda t, y: rober_numba(t, y, k1, k2, k3),
                [t0, t1],
                y0,
                method="LSODA",
                rtol=rtol,
                atol=atol,
                t_eval=None,
                dense_output=False,
                first_step=1e-4,
                max_step=t1/2,
            )
        
        if not sol.success:
            # Final fallback to Radau
            sol = solve_ivp(
                lambda t, y: rober_numba(t, y, k1, k2, k3),
                [t0, t1],
                y0,
                method="Radau",
                rtol=rtol,
                atol=atol,
                t_eval=None,
                dense_output=False,
                first_step=1e-4,
                max_step=t1/2,
            )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()