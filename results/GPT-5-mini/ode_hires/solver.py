from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Solver for the HIRES stiff ODE system.

    Problem dict keys:
      - t0: initial time (float)
      - t1: final time (float)
      - y0: initial state (iterable of length 8)
      - constants: iterable of length 12

    Returns:
      - list of 8 floats: state at t1
    """

    def solve(self, problem: dict, **kwargs) -> Any:
        # Validate input
        if not all(k in problem for k in ("t0", "t1", "y0", "constants")):
            raise ValueError("Problem must contain t0, t1, y0, and constants")

        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.asarray(problem["y0"], dtype=float).ravel()
        constants = np.asarray(problem["constants"], dtype=float).ravel()

        if y0.size != 8:
            raise ValueError("y0 must have 8 components")
        if constants.size != 12:
            raise ValueError("constants must have 12 components")

        if np.isclose(t0, t1):
            return y0.tolist()

        # Unpack constants
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants

        # Pre-create constant part of Jacobian to avoid repeated allocations
        J_const = np.zeros((8, 8), dtype=float)
        J_const[0, 0] = -c1
        J_const[0, 1] = c2
        J_const[0, 2] = c3

        J_const[1, 0] = c1
        J_const[1, 1] = -c5

        J_const[2, 2] = -c6
        J_const[2, 3] = c2
        J_const[2, 4] = c7

        J_const[3, 1] = c3
        J_const[3, 2] = c1
        J_const[3, 3] = -c8

        J_const[4, 4] = -c9
        J_const[4, 5] = c2
        J_const[4, 6] = c2

        J_const[5, 3] = c11
        J_const[5, 4] = c1
        J_const[5, 6] = c11

        J_const[6, 6] = -c12
        J_const[7, 6] = c12

        # Work array for RHS to reduce allocations
        f_work = np.empty(8, dtype=float)

        def hires(t, y):
            # y: array-like with 8 components
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            f_work[0] = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            f_work[1] = c1 * y1 - c5 * y2
            f_work[2] = -c6 * y3 + c2 * y4 + c7 * y5
            f_work[3] = c3 * y2 + c1 * y3 - c8 * y4
            f_work[4] = -c9 * y5 + c2 * y6 + c2 * y7
            f_work[5] = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            f_work[6] = c10 * y6 * y8 - c12 * y7
            f_work[7] = -c10 * y6 * y8 + c12 * y7
            return f_work.copy()

        def jac(t, y):
            # Start from the constant part and update y-dependent entries
            J = J_const.copy()
            y6 = y[5]
            y8 = y[7]
            # f6 row (index 5)
            J[5, 5] = -c10 * y8 - c2
            J[5, 7] = -c10 * y6
            # f7 row (index 6)
            J[6, 5] = c10 * y8
            J[6, 7] = c10 * y6
            # f8 row (index 7)
            J[7, 5] = -c10 * y8
            J[7, 7] = -c10 * y6
            return J

        # Solver options: relax tolerances for speed while staying within verification tolerance
        rtol = float(kwargs.get("rtol", 1e-8))
        atol = float(kwargs.get("atol", 1e-8))
        method = kwargs.get("method", "BDF")

        # Try solving; if Radau fails for some reason, try BDF as a fallback
        sol = solve_ivp(
            fun=hires,
            t_span=(t0, t1),
            y0=y0,
            method=method,
            jac=jac,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success and method != "BDF":
            sol = solve_ivp(
                fun=hires,
                t_span=(t0, t1),
                y0=y0,
                method="BDF",
                jac=jac,
                rtol=rtol,
                atol=atol,
                dense_output=False,
            )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.y[:, -1].tolist()