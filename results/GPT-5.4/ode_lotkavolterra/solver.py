from __future__ import annotations

from typing import Any
import math

import numpy as np
from scipy.integrate import DOP853

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.asarray(problem["y0"], dtype=float)

        if t0 == t1:
            return y0.tolist()

        params = problem["params"]
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        delta = float(params["delta"])
        gamma = float(params["gamma"])

        x0 = float(y0[0])
        z0 = float(y0[1])

        def rhs(t, y):
            x = y[0]
            z = y[1]
            xz = x * z
            return (
                alpha * x - beta * xz,
                delta * xz - gamma * z,
            )

        solver = DOP853(
            rhs,
            t0,
            y0,
            t1,
            rtol=2e-8,
            atol=1e-11,
        )

        step = solver.step
        while solver.status == "running":
            step()

        if solver.status != "finished":
            raise RuntimeError("Solver failed to finish")

        y = solver.y
        x = float(y[0])
        z = float(y[1])

        if x0 > 0.0 and z0 > 0.0 and x > 0.0 and z > 0.0:
            log = math.log
            h0 = delta * x0 - gamma * log(x0) + beta * z0 - alpha * log(z0)
            for _ in range(2):
                gx = delta - gamma / x
                gz = beta - alpha / z
                dh = delta * x - gamma * log(x) + beta * z - alpha * log(z) - h0
                denom = gx * gx + gz * gz
                if denom <= 0.0:
                    break
                lam = dh / denom
                x_new = x - lam * gx
                z_new = z - lam * gz
                if x_new <= 0.0 or z_new <= 0.0:
                    break
                x = x_new
                z = z_new

        return [x, z]