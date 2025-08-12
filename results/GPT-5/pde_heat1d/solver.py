from __future__ import annotations

from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solve the semi-discrete 1D heat equation with Dirichlet BCs using RK45,
        matching the reference implementation's numerical pathway.
        """
        y0 = np.array(problem["y0"])
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]

        n = y0.size
        if n == 0:
            return []

        # Preallocate padded buffer to avoid per-call allocations from np.pad
        u_padded = np.empty(n + 2, dtype=y0.dtype)
        c = alpha / (dx * dx)

        def heat_equation(_t: float, u: np.ndarray) -> np.ndarray:
            # Manually pad with zeros (Dirichlet BCs)
            u_padded[0] = 0.0
            u_padded[-1] = 0.0
            u_padded[1:-1] = u

            # Compute second derivative using ufuncs with out= to minimize temporaries:
            # du = u_padded[2:] - 2*u_padded[1:-1] + u_padded[:-2]
            du = np.empty_like(u)
            np.subtract(u_padded[2:], u_padded[1:-1], out=du)       # du = a - b
            np.subtract(du, u_padded[1:-1], out=du)                 # du = du - b (=> a - 2b)
            np.add(du, u_padded[:-2], out=du)                       # du = du + c

            # Scale by coefficient
            du *= c
            return du

        # Match reference tolerances and method
        rtol = 1e-6
        atol = 1e-6
        method = "RK45"

        sol = solve_ivp(
            heat_equation,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()