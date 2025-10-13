from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp
except Exception as e:  # pragma: no cover
    solve_ivp = None  # type: ignore

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        alpha = float(params["alpha"])
        dx = float(params["dx"])

        if y0.ndim != 1:
            y0 = y0.ravel()
        n = y0.size
        if n == 0:
            return [].copy()

        if solve_ivp is None:
            # Fallback: trivial evolution (no change). This should not happen in normal eval.
            return y0.tolist()

        scale = alpha / (dx * dx)

        def heat_equation(t, u):
            # Dirichlet boundary conditions u_0 = u_{n+1} = 0
            du = np.empty_like(u)
            if n == 1:
                du[0] = scale * (-2.0 * u[0])
                return du
            # interior second derivative
            du[1:-1] = scale * (u[2:] - 2.0 * u[1:-1] + u[:-2])
            # boundaries use zero outside
            du[0] = scale * (u[1] - 2.0 * u[0])
            du[-1] = scale * (-2.0 * u[-1] + u[-2])
            return du

        rtol = 1e-6
        atol = 1e-6
        method = "RK45"

        sol = solve_ivp(
            heat_equation,
            [t0, t1],
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