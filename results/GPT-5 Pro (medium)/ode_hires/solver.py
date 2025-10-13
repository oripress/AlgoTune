from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp, ode

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the HIRES stiff ODE system from t0 to t1 with given initial conditions and constants.

        Parameters
        ----------
        problem : dict
            {
              "t0": float,
              "t1": float,
              "y0": list[8 floats],
              "constants": list[12 floats]
            }

        Returns
        -------
        list[float]
            The state y(t1) as a list of 8 floats.
        """
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        c = np.asarray(problem["constants"], dtype=float)

        # Unpack constants once for fast local access in inner functions
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = c

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3, y4, y5, y6, y7, y8 = y

            f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            f2 = c1 * y1 - c5 * y2
            f3 = -c6 * y3 + c2 * y4 + c7 * y5
            f4 = c3 * y2 + c1 * y3 - c8 * y4
            f5 = -c9 * y5 + c2 * y6 + c2 * y7
            tmp = c10 * y6 * y8
            f6 = -tmp + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            f7 = tmp - c12 * y7
            f8 = -tmp + c12 * y7

            return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=float)

        def jac(_t: float, y: np.ndarray) -> np.ndarray:
            # Analytical Jacobian for the HIRES system
            y1, y2, y3, y4, y5, y6, y7, y8 = y

            J = np.zeros((8, 8), dtype=float)

            # df1/dy*
            J[0, 0] = -c1
            J[0, 1] = c2
            J[0, 2] = c3

            # df2/dy*
            J[1, 0] = c1
            J[1, 1] = -c5

            # df3/dy*
            J[2, 2] = -c6
            J[2, 3] = c2
            J[2, 4] = c7

            # df4/dy*
            J[3, 1] = c3
            J[3, 2] = c1
            J[3, 3] = -c8

            # df5/dy*
            J[4, 4] = -c9
            J[4, 5] = c2
            J[4, 6] = c2

            # df6/dy*
            J[5, 3] = c11
            J[5, 4] = c1
            J[5, 5] = -c2 - c10 * y8
            J[5, 6] = c11
            J[5, 7] = -c10 * y6

            # df7/dy*
            J[6, 5] = c10 * y8
            J[6, 6] = -c12
            J[6, 7] = c10 * y6

            # df8/dy*
            J[7, 5] = -c10 * y8
            J[7, 6] = c12
            J[7, 7] = -c10 * y6

            return J

        # Try VODE-BDF with analytic Jacobian (often faster for small stiff systems)
        try:
            r = ode(rhs, jac)
            # 'vode' with 'bdf' method and analytic Jacobian
            r.set_integrator(
                "vode",
                method="bdf",
                with_jacobian=True,
                rtol=1e-10,
                atol=1e-9,
                nsteps=1_000_000,  # allow many internal steps for long intervals
                order=5,
            )
            r.set_initial_value(y0, t0)
            y_final = r.integrate(t1)
            if not r.successful():
                raise RuntimeError("VODE integration unsuccessful")
            return np.asarray(y_final, dtype=float).tolist()
        except Exception:
            # Fallback to solve_ivp Radau with analytic Jacobian
            sol = solve_ivp(
                rhs,
                (t0, t1),
                y0,
                method="Radau",
                jac=jac,
                rtol=1e-10,
                atol=1e-9,
            )
            if not sol.success:
                raise RuntimeError(f"Solver failed: {sol.message}")
            return sol.y[:, -1].tolist()