from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mu = float(problem["mu"])
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])

        def vdp(t, y):
            x, v = y
            return [v, mu * ((1.0 - x*x) * v - x)]

        def jac(t, y):
            x, v = y
            return [[0.0, 1.0],
                    [-mu * (2.0 * x * v + 1.0), mu * (1.0 - x*x)]]

        sol = solve_ivp(
            vdp,
            (t0, t1),
            y0,
            method="LSODA",
            jac=jac,
            rtol=1e-8,
            atol=1e-9,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return [float(sol.y[0, -1]), float(sol.y[1, -1])]