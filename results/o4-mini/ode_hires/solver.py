from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        # Unpack inputs
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        c = problem["constants"]
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = c

        # HIRES ODE system
        def hires(t, y):
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            return [
                -c1 * y1 + c2 * y2 + c3 * y3 + c4,
                c1 * y1 - c5 * y2,
                -c6 * y3 + c2 * y4 + c7 * y5,
                c3 * y2 + c1 * y3 - c8 * y4,
                -c9 * y5 + c2 * y6 + c2 * y7,
                -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7,
                c10 * y6 * y8 - c12 * y7,
                -c10 * y6 * y8 + c12 * y7,
            ]

        # Analytic Jacobian for stiff solver
        def jac(t, y):
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            J = np.zeros((8, 8), dtype=float)
            J[0, 0] = -c1; J[0, 1] = c2; J[0, 2] = c3
            J[1, 0] = c1;  J[1, 1] = -c5
            J[2, 2] = -c6; J[2, 3] = c2; J[2, 4] = c7
            J[3, 1] = c3;  J[3, 2] = c1; J[3, 3] = -c8
            J[4, 4] = -c9; J[4, 5] = c2; J[4, 6] = c2
            J[5, 3] = c11; J[5, 4] = c1; J[5, 5] = -c10 * y8 - c2; J[5, 6] = c11; J[5, 7] = -c10 * y6
            J[6, 5] = c10 * y8; J[6, 6] = -c12; J[6, 7] = c10 * y6
            J[7, 5] = -c10 * y8; J[7, 6] = c12; J[7, 7] = -c10 * y6
            return J

        # Integrate with LSODA + Jacobian for accuracy and speed
        sol = solve_ivp(
            hires,
            (t0, t1),
            y0,
            method="LSODA",
            jac=jac,
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            # Fallback to Radau if LSODA fails
            sol = solve_ivp(
                hires,
                (t0, t1),
                y0,
                method="Radau",
                jac=jac,
                rtol=1e-8,
                atol=1e-10,
            )
            if not sol.success:
                raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Return final state
        return sol.y[:, -1].tolist()