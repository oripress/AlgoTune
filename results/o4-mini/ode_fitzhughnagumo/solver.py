import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the FitzHugh-Nagumo ODE using SciPy's RK45 integrator.
        """
        # Unpack problem data
        y0 = problem["y0"]
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        I = params["I"]

        # Define the system of ODEs
        def f(t, y):
            v, w = y
            return [v - (v**3) / 3.0 - w + I, a * (b * v - c * w)]

        # Integrate from t0 to t1 without intermediate outputs
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            vectorized=False,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        # Return final state [v, w]
        return sol.y[:, -1].tolist()