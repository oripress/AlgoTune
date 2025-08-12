import numpy as np
from typing import Any
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Solve the Brusselator model efficiently."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        A = params["A"]
        B = params["B"]
        
        def brusselator(t, y):
            X, Y = y
            dX_dt = A + X**2 * Y - (B + 1) * X
            dY_dt = B * X - X**2 * Y
            return np.array([dX_dt, dY_dt])
        
        # Solve the ODE with optimized settings
        # Solve the ODE with optimized settings
        # Solve the ODE with optimized settings
        sol = solve_ivp(
            brusselator,
            [t0, t1],
            y0,
            method="DOP853",
            rtol=1e-8,
            atol=1e-8,
        )
        
        return sol.y[:, -1]