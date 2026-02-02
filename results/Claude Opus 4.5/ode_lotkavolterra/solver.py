from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def __init__(self):
        pass

    def solve(self, problem, **kwargs) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        gamma = params["gamma"]

        def lotka_volterra(t, y):
            x, pred = y[0], y[1]
            dx_dt = alpha * x - beta * x * pred
            dy_dt = delta * x * pred - gamma * pred
            return [dx_dt, dy_dt]

        rtol = 1e-10
        atol = 1e-10

        # Try LSODA - adaptive method that automatically switches between stiff and non-stiff
        sol = solve_ivp(
            lotka_volterra,
            [t0, t1],
            y0,
            method="LSODA",
            rtol=rtol,
            atol=atol,
        )

        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")