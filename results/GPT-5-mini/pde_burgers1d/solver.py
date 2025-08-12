from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solve the 1D viscous Burgers equation using method-of-lines:
        - upwind scheme for the advection term
        - central differences for the diffusion term
        Dirichlet BCs (u=0 at both domain boundaries).
        Returns the solution at final time as a list of floats.
        """
        if not isinstance(problem, dict):
            raise ValueError("problem must be a dict")
        if "y0" not in problem:
            raise ValueError("problem must contain 'y0'")

        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem.get("t0", 0.0))
        t1 = float(problem.get("t1", 0.5))
        params = problem.get("params", {}) or {}
        nu = float(params.get("nu", 0.005))
        dx = float(params.get("dx", 1.0))

        if dx <= 0:
            raise ValueError("dx must be positive")

        def burgers_rhs(t, u):
            # Pad for Dirichlet BCs (u=0 at boundaries)
            u_padded = np.pad(u, 1, mode="constant", constant_values=0.0)

            # Diffusion: central second difference
            diffusion = (u_padded[2:] - 2.0 * u_padded[1:-1] + u_padded[:-2]) / (dx * dx)

            # Advection: upwind scheme based on sign of u
            u_center = u_padded[1:-1]
            du_dx_forward = (u_padded[2:] - u_padded[1:-1]) / dx
            du_dx_backward = (u_padded[1:-1] - u_padded[:-2]) / dx
            advection = np.where(u_center >= 0.0, u_center * du_dx_backward, u_center * du_dx_forward)

            return -advection + nu * diffusion

        # Solver options (match reference tolerances by default)
        rtol = float(kwargs.pop("rtol", 1e-6))
        atol = float(kwargs.pop("atol", 1e-6))
        method = kwargs.pop("method", "RK45")

        sol = solve_ivp(
            burgers_rhs,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.y[:, -1].tolist()