import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract all parameters at once
        t0, t1 = problem["t0"], problem["t1"]
        y0 = problem["y0"]
        p = problem["params"]
        alpha, beta, delta, gamma = p["alpha"], p["beta"], p["delta"], p["gamma"]
        
        # Precomputed parameters for efficiency
        def lotka_volterra(t, y):
            x, y_pred = y
            dx_dt = x * (alpha - beta * y_pred)
            dy_dt = y_pred * (delta * x - gamma)
            return [dx_dt, dy_dt]
        
        # Optimized LSODA solver
        sol = solve_ivp(
            lotka_volterra,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-10,
            atol=1e-10,
            dense_output=False,
            vectorized=False
        )
        
        # Directly return the last state without array conversion
        return sol.y[:, -1].tolist()