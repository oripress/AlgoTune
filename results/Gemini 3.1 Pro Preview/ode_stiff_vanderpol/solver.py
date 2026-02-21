import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict, **kwargs) -> list[float]:
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])

        def vdp(t, y):
            return [y[1], mu * ((1 - y[0]**2) * y[1] - y[0])]

        def jac(t, y):
            return [[0, 1],
                    [-mu * (2 * y[0] * y[1] + 1), mu * (1 - y[0]**2)]]

        sol = solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method="BDF",
            jac=jac,
            rtol=1e-8,
            atol=1e-9,
        )

        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")