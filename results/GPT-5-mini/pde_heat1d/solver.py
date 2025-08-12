from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solve the 1D heat equation using the same approach as the reference:
        method of lines with central finite differences in space and solve_ivp (RK45).
        Returns the solution at final time t1 as a list of floats.
        """
        # Extract inputs
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem.get("params", {})
        alpha = float(params["alpha"])
        dx = float(params["dx"])

        # Right-hand side for ODE system du/dt = alpha * u_xx with Dirichlet BCs (0 at ends)
        def heat_equation(t, u):
            # pad with zeros for Dirichlet boundaries
            u_padded = np.empty(u.size + 2, dtype=float)
            u_padded[0] = 0.0
            u_padded[-1] = 0.0
            u_padded[1:-1] = u
            u_xx = (u_padded[2:] - 2.0 * u_padded[1:-1] + u_padded[:-2]) / (dx * dx)
            return alpha * u_xx

        # Solver tolerances (match reference)
        rtol = 1e-6
        atol = 1e-6

        sol = solve_ivp(
            heat_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()