import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])
        N = len(y0)
        
        # Pre-compute indices once
        indices = np.arange(N)
        ip1 = (indices + 1) % N
        im1 = (indices - 1) % N
        im2 = (indices - 2) % N
        
        def lorenz96(t, x):
            # Direct vectorized computation without creating new arrays
            return (x[ip1] - x[im2]) * x[im1] - x + F
        
        # Solve the ODE with optimized parameters
        sol = solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")