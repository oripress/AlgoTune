from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        # Extract parameters
        mu = problem["mu"]
        y0 = problem["y0"]
        t0 = problem["t0"]
        t1 = problem["t1"]
        
        # Define the Van der Pol ODE system
        def vdp(t, y):
            x, v = y
            dx_dt = v
            dv_dt = mu * ((1 - x**2) * v - x)
            return [dx_dt, dv_dt]
        
        # The Van der Pol equation is stiff, so we use LSODA which automatically switches methods
        sol = solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-6,
            atol=1e-8,
            dense_output=False
        )
        # Return the final state as a list
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")