import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
from numba import njit

@njit(fastmath=True, cache=True)
def fitzhugh_nagumo_jit(t, y, a, b, c, I):
    v = y[0]
    w = y[1]
    v3 = v * v * v
    dv_dt = v - v3 / 3.0 - w + I
    dw_dt = a * (b * v - c * w)
    return np.array([dv_dt, dw_dt])

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        
        # Pre-extract parameters
        a = problem["params"]["a"]
        b = problem["params"]["b"]
        c = problem["params"]["c"]
        I = problem["params"]["I"]
        
        # Wrapper for scipy
        def fitzhugh_nagumo(t, y):
            return fitzhugh_nagumo_jit(t, y, a, b, c, I)
        
        sol = solve_ivp(
            fitzhugh_nagumo,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()