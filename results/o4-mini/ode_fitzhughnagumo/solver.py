import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract initial conditions and parameters
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])

        # FitzHugh-Nagumo equations
        def rhs(t, y):
            v, w = y
            dv_dt = v - (v**3) / 3.0 - w + I
            dw_dt = a * (b * v - c * w)
            return [dv_dt, dw_dt]

        # Integrate with RK45 matching the reference solver
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return final state [v, w]
        return sol.y[:, -1].tolist()