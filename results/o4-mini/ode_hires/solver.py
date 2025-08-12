from typing import Any, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        # Unpack inputs
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = problem["constants"]

        # Trivial case
        if t1 == t0:
            return y0.tolist()

        # ODE RHS
        def fun(t, y):
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            return np.array([
                -c1*y1 + c2*y2 + c3*y3 + c4,
                c1*y1 - c5*y2,
                -c6*y3 + c2*y4 + c7*y5,
                c3*y2 + c1*y3 - c8*y4,
                -c9*y5 + c2*y6 + c2*y7,
                -c10*y6*y8 + c11*y4 + c1*y5 - c2*y6 + c11*y7,
                c10*y6*y8 - c12*y7,
                -c10*y6*y8 + c12*y7,
            ])

        # Analytic Jacobian
        def jac(t, y):
            y6 = y[5]; y8 = y[7]
            return np.array([
                [-c1,       c2,    c3,    0.0,   0.0,          0.0,    0.0,       0.0],
                [ c1,      -c5,    0.0,    0.0,   0.0,          0.0,    0.0,       0.0],
                [ 0.0,      0.0,  -c6,     c2,     c7,         0.0,    0.0,       0.0],
                [ 0.0,      c3,    c1,    -c8,    0.0,         0.0,    0.0,       0.0],
                [ 0.0,      0.0,   0.0,     0.0,   -c9,         c2,     c2,       0.0],
                [ 0.0,      0.0,   0.0,    c11,    c1,   -(c10*y8 + c2), c11, -c10*y6],
                [ 0.0,      0.0,   0.0,     0.0,   0.0,    c10*y8,    -c12,  c10*y6],
                [ 0.0,      0.0,   0.0,     0.0,   0.0,   -c10*y8,     c12, -c10*y6],
            ])

        # Integrate using LSODA (ODEPACK)
        sol = solve_ivp(
            fun, (t0, t1), y0,
            method='LSODA',
            jac=jac,
            rtol=1e-10,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y[:, -1].tolist()