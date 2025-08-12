import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
import numba

class Solver:
    def __init__(self):
        # Pre-compile the numba function for speed
        @numba.njit
        def vdp_numba(t, y, mu):
            x = y[0]
            v = y[1]
            dx_dt = v
            dv_dt = mu * ((1.0 - x*x) * v - x)
            result = np.empty(2)
            result[0] = dx_dt
            result[1] = dv_dt
            return result
        
        self.vdp_numba = vdp_numba
    
    def solve(self, problem: dict[str, np.ndarray | float], **kwargs) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])
        
        # Create a wrapper that uses the pre-compiled numba function
        def vdp(t, y):
            return self.vdp_numba(t, y, mu)
        
        # Use LSODA which can automatically switch between stiff and non-stiff methods
        sol = solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method="LSODA",
            rtol=1e-8,
            atol=1e-9,
            t_eval=None,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")