import numpy as np
from scipy.integrate import odeint

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])

        def vdp(y, t):
            x, v = y[0], y[1]
            return [v, mu * ((1 - x*x) * v - x)]

        def jac(y, t):
            x, v = y[0], y[1]
            return [[0.0, 1.0],
                    [mu * (-2.0 * x * v - 1.0), mu * (1.0 - x*x)]]

        sol = odeint(
            vdp,
            y0,
            [t0, t1],
            Dfun=jac,
            col_deriv=False,
            rtol=1e-6,
            atol=1e-8,
            mxstep=100000,
        )

        return [sol[-1, 0], sol[-1, 1]]