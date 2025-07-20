import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract initial conditions and parameters
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        alpha = float(params["alpha"])
        beta  = float(params["beta"])
        delta = float(params["delta"])
        gamma = float(params["gamma"])

        # Define Lotka-Volterra system
        def lotka_volterra(t, y):
            x, y_p = y
            return [alpha * x - beta * x * y_p,
                    delta * x * y_p - gamma * y_p]

        sol = solve_ivp(
            lotka_volterra,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-10,
            atol=1e-10
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return final populations
        result = sol.y[:, -1]
        return result.tolist()