from __future__ import annotations

from typing import Any, List

import numpy as np
from scipy.integrate import solve_ivp, ode, odeint

class Solver:
    def solve(self, problem: dict, **kwargs) -> List[float]:
        """
        Solve the HIRES stiff ODE system using fast ODEPACK integrators with banded analytic Jacobian.
        Strategy:
          1) LSODA (odeint) with banded Jacobian
          2) VODE (BDF) with banded Jacobian
          3) Fallback to solve_ivp BDF/Radau with dense Jacobian

        Parameters
        ----------
        problem : dict
            {
                "t0": float,
                "t1": float,
                "y0": list[float] of length 8,
                "constants": list[float] of length 12
            }

        Returns
        -------
        List[float]
            Final state y(t1) as a list of 8 floats.
        """
        # Extract problem data
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.asarray(problem["y0"], dtype=float)
        constants = problem["constants"]
        # Unpack constants once for speed
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = map(float, constants)

        # RHS function (shared by ODEPACK and solve_ivp)
        def fun(t: float, y: np.ndarray):
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            f2 = c1 * y1 - c5 * y2
            f3 = -c6 * y3 + c2 * y4 + c7 * y5
            f4 = c3 * y2 + c1 * y3 - c8 * y4
            f5 = -c9 * y5 + c2 * y6 + c2 * y7
            f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            f7 = c10 * y6 * y8 - c12 * y7
            f8 = -c10 * y6 * y8 + c12 * y7
            return (f1, f2, f3, f4, f5, f6, f7, f8)

        # For odeint (y, t) signature; return tuple for minimal overhead
        def fun_odeint(y: np.ndarray, t: float):
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            f2 = c1 * y1 - c5 * y2
            f3 = -c6 * y3 + c2 * y4 + c7 * y5
            f4 = c3 * y2 + c1 * y3 - c8 * y4
            f5 = -c9 * y5 + c2 * y6 + c2 * y7
            f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            f7 = c10 * y6 * y8 - c12 * y7
            f8 = -c10 * y6 * y8 + c12 * y7
            return (f1, f2, f3, f4, f5, f6, f7, f8)

        # Dense Jacobian for solve_ivp fallbacks
        def jac_dense(t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            J = np.zeros((8, 8), dtype=float)
            # Row 1
            J[0, 0] = -c1
            J[0, 1] = c2
            J[0, 2] = c3
            # Row 2
            J[1, 0] = c1
            J[1, 1] = -c5
            # Row 3
            J[2, 2] = -c6
            J[2, 3] = c2
            J[2, 4] = c7
            # Row 4
            J[3, 1] = c3
            J[3, 2] = c1
            J[3, 3] = -c8
            # Row 5
            J[4, 4] = -c9
            J[4, 5] = c2
            J[4, 6] = c2
            # Row 6
            J[5, 3] = c11
            J[5, 4] = c1
            J[5, 5] = -c10 * y8 - c2
            J[5, 6] = c11
            J[5, 7] = -c10 * y6
            # Row 7
            J[6, 5] = c10 * y8
            J[6, 6] = -c12
            J[6, 7] = c10 * y6
            # Row 8
            J[7, 5] = -c10 * y8
            J[7, 6] = c12
            J[7, 7] = -c10 * y6
            return J

        # Precompute constant banded Jacobian template (ml=2, mu=2)
        ab_template = np.zeros((5, 8), dtype=float)
        # Fill constant entries once
        # Row 0
        ab_template[2 + 0 - 0, 0] = -c1         # J00
        ab_template[2 + 0 - 1, 1] = c2          # J01
        ab_template[2 + 0 - 2, 2] = c3          # J02
        # Row 1
        ab_template[2 + 1 - 0, 0] = c1          # J10
        ab_template[2 + 1 - 1, 1] = -c5         # J11
        # Row 2
        ab_template[2 + 2 - 2, 2] = -c6         # J22
        ab_template[2 + 2 - 3, 3] = c2          # J23
        ab_template[2 + 2 - 4, 4] = c7          # J24
        # Row 3
        ab_template[2 + 3 - 1, 1] = c3          # J31
        ab_template[2 + 3 - 2, 2] = c1          # J32
        ab_template[2 + 3 - 3, 3] = -c8         # J33
        # Row 4
        ab_template[2 + 4 - 4, 4] = -c9         # J44
        ab_template[2 + 4 - 5, 5] = c2          # J45
        ab_template[2 + 4 - 6, 6] = c2          # J46
        # Row 5
        ab_template[2 + 5 - 3, 3] = c11         # J53
        ab_template[2 + 5 - 4, 4] = c1          # J54
        # J55, J57 depend on y8, y6
        ab_template[2 + 5 - 6, 6] = c11         # J56 constant
        # Row 6
        # J65 depends on y8
        ab_template[2 + 6 - 6, 6] = -c12        # J66 constant
        # J67 depends on y6
        # Row 7
        # J75 depends on y8
        ab_template[2 + 7 - 6, 6] = c12         # J76 constant
        # J77 depends on y6

        # Working banded Jacobian to mutate dynamic entries only
        ab_work = ab_template.copy()

        # Helper to update dynamic entries in the banded Jacobian
        def _update_ab(y6: float, y8: float):
            # J55: ab[2,5]
            ab_work[2, 5] = -c10 * y8 - c2
            # J57: ab[0,7]
            ab_work[0, 7] = -c10 * y6
            # J65: ab[3,5]
            ab_work[3, 5] = c10 * y8
            # J67: ab[1,7]
            ab_work[1, 7] = c10 * y6
            # J75: ab[4,5]
            ab_work[4, 5] = -c10 * y8
            # J77: ab[2,7]
            ab_work[2, 7] = -c10 * y6

        # Banded Jacobian for ODEPACK (ml=2, mu=2)
        def jac_banded(t: float, y: np.ndarray) -> np.ndarray:
            y6, y8 = y[5], y[7]
            _update_ab(y6, y8)
            return ab_work

        # For odeint (y, t)
        def jac_banded_odeint(y: np.ndarray, t: float) -> np.ndarray:
            y6, y8 = y[5], y[7]
            _update_ab(y6, y8)
            return ab_work

        # Tolerances: tuned for accuracy and speed
        rtol = 1e-8
        atol = 1e-10

        # Attempt 1: LSODA via odeint with banded Jacobian
        try:
            tspan = np.array([t0, t1], dtype=float)
            y_out = odeint(
                func=fun_odeint,
                y0=y0,
                t=tspan,
                Dfun=jac_banded_odeint,
                col_deriv=0,
                ml=2,
                mu=2,
                rtol=rtol,
                atol=atol,
                hmax=0.0,        # no explicit cap
                mxstep=1_000_000,
                printmessg=False,
                tfirst=False,
            )
            y_final = y_out[-1]
            if np.all(np.isfinite(y_final)):
                return y_final.tolist()
        except Exception:
            # Fall through to next attempt
            pass

        # Attempt 2: VODE (BDF) with banded Jacobian
        try:
            integrator = ode(fun, jac_banded)
            integrator.set_integrator(
                "vode",
                method="bdf",
                with_jacobian=True,
                rtol=rtol,
                atol=atol,
                nsteps=1_000_000,
                order=5,
                lband=2,
                uband=2,
            )
            integrator.set_initial_value(y0, t0)
            y_final = integrator.integrate(t1)
            if integrator.successful():
                return y_final.tolist()
        except Exception:
            # Fall through to solve_ivp fallback
            pass

        # Fallback 1: BDF with dense analytic Jacobian
        sol = solve_ivp(
            fun,
            (t0, t1),
            y0,
            method="BDF",
            jac=jac_dense,
            rtol=rtol,
            atol=atol,
            dense_output=False,
            vectorized=False,
        )

        if not sol.success:
            # Fallback 2: Radau with dense analytic Jacobian and tighter tolerances
            sol = solve_ivp(
                fun,
                (t0, t1),
                y0,
                method="Radau",
                jac=jac_dense,
                rtol=1e-9,
                atol=1e-10,
                dense_output=False,
                vectorized=False,
            )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()