import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Unpack problem
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]

        # ODE right‐hand side
        def heat_rhs(t, u):
            # Dirichlet zero‐boundary at both ends
            u_p = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)
            # second derivative
            u_xx = (u_p[2:] - 2*u_p[1:-1] + u_p[:-2]) / (dx*dx)
            return alpha * u_xx

        # Integrate from t0 to t1
        sol = solve_ivp(
            heat_rhs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        # Return final interior temperatures
        return sol.y[:, -1].tolist()