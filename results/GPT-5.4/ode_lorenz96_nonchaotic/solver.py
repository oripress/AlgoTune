from typing import Any

import numpy as np
from scipy.integrate import RK45

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        f_force = float(problem["F"])

        n = y0.size
        if n == 0 or t0 == t1:
            return y0.tolist()

        if n == 1:
            return [float((y0[0] - f_force) * np.exp(-(t1 - t0)) + f_force)]

        if n == 2:
            def lorenz96(_t: float, x: np.ndarray) -> np.ndarray:
                x0 = x[0]
                x1 = x[1]
                return np.array(
                    [
                        (x1 - x0) * x1 - x0 + f_force,
                        (x0 - x1) * x0 - x1 + f_force,
                    ],
                    dtype=np.float64,
                )
        else:
            def lorenz96(_t: float, x: np.ndarray) -> np.ndarray:
                dx = np.empty_like(x)
                dx[0] = (x[1] - x[n - 2]) * x[n - 1] - x[0] + f_force
                dx[1] = (x[2] - x[n - 1]) * x[0] - x[1] + f_force
                if n > 3:
                    dx[2:-1] = (x[3:] - x[:-3]) * x[1:-2] - x[2:-1] + f_force
                dx[n - 1] = (x[0] - x[n - 3]) * x[n - 2] - x[n - 1] + f_force
                return dx

        solver = RK45(lorenz96, t0, y0, t1, rtol=1e-8, atol=1e-8)
        while solver.status == "running":
            solver.step()
        if solver.status == "failed":
            raise RuntimeError("Solver failed")
        return solver.y.tolist()