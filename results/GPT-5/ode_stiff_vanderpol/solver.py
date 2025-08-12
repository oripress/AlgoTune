from typing import Any

import math
import numpy as np
from scipy.integrate import solve_ivp, ode, odeint

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solve the stiff Van der Pol oscillator:
            x' = v
            v' = mu * ((1 - x^2) * v - x)

        problem:
            - mu: float
            - y0: list/array of length 2
            - t0: float
            - t1: float

        Returns:
            list[float]: [x(t1), v(t1)]
        """
        mu = float(problem["mu"])
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])

        if y0.shape != (2,):
            y0 = np.asarray(y0, dtype=float).reshape(-1)
            if y0.shape != (2,):
                raise ValueError("y0 must be a length-2 vector.")

        if t1 == t0:
            return y0.tolist()

        # Closed-form fast path for the linear case mu == 0
        if mu == 0.0:
            dt = t1 - t0
            c, s = math.cos(dt), math.sin(dt)
            x0, v0 = float(y0[0]), float(y0[1])
            x1 = x0 * c + v0 * s
            v1 = -x0 * s + v0 * c
            return [x1, v1]

        # Tolerances
        rtol = 1e-8
        atol = 1e-9

        # Primary fast path: odeint (LSODA) with analytic Jacobian
        dt = abs(t1 - t0)

        def f_odeint(y, t, mu_param):
            x = y[0]
            v = y[1]
            return (v, mu_param * ((1.0 - x * x) * v - x))

        def j_odeint(y, t, mu_param):
            x = y[0]
            v = y[1]
            return ((0.0, 1.0), (mu_param * (-2.0 * x * v - 1.0), mu_param * (1.0 - x * x)))

        try:
            tspan = (t0, t1)
            y_arr = odeint(
                f_odeint,
                y0,
                tspan,
                args=(mu,),
                Dfun=j_odeint,
                atol=atol,
                rtol=rtol,
                mxstep=2_000_000,
                hmax=dt if dt > 0.0 else 0.0,
                full_output=False,
                tfirst=False,
            )
            y_end = y_arr[-1]
            if np.all(np.isfinite(y_end)) and y_end.shape == (2,):
                return y_end.tolist()
        except Exception:
            # Fall through to alternative solvers
            pass

        # Secondary fast path: scipy.integrate.ode with LSODA / VODE-BDF
        try:
            def f_ode(t: float, y: np.ndarray, mu_local: float):
                x, v = y
                return (v, mu_local * ((1.0 - x * x) * v - x))

            def j_ode(t: float, y: np.ndarray, mu_local: float):
                x, v = y
                return ((0.0, 1.0), (mu_local * (-2.0 * x * v - 1.0), mu_local * (1.0 - x * x)))

            integrator = ode(f_ode).set_integrator(
                "lsoda",
                rtol=rtol,
                atol=atol,
                mxstep=1_000_000,
                hmax=dt if dt > 0.0 else 0.0,
            )
            integrator.set_f_params(mu)
            integrator.set_initial_value(y0, t0)
            y_end = integrator.integrate(t1)
            if integrator.successful() and np.all(np.isfinite(y_end)) and y_end.shape == (2,):
                return y_end.tolist()

            # If LSODA path failed or was not successful, try VODE-BDF with analytic Jacobian
            integrator = ode(f_ode, j_ode).set_integrator(
                "vode", method="bdf", with_jacobian=True, rtol=rtol, atol=atol, nsteps=200000
            )
            integrator.set_f_params(mu)
            integrator.set_jac_params(mu)
            integrator.set_initial_value(y0, t0)
            y_end = integrator.integrate(t1)
            if integrator.successful() and np.all(np.isfinite(y_end)) and y_end.shape == (2,):
                return y_end.tolist()
        except Exception:
            # Fall back to solve_ivp if the low-level integrator fails
            pass

        # Final fallback: solve_ivp with Radau (stiff) and analytic Jacobian
        def f_ivp(t: float, y: np.ndarray):
            x, v = y
            return (v, mu * ((1.0 - x * x) * v - x))

        def j_ivp(t: float, y: np.ndarray):
            x, v = y
            return ((0.0, 1.0), (mu * (-2.0 * x * v - 1.0), mu * (1.0 - x * x)))

        sol = solve_ivp(
            f_ivp,
            (t0, t1),
            y0,
            method="Radau",
            jac=j_ivp,
            rtol=rtol,
            atol=atol,
            t_eval=(t1,),
            dense_output=False,
            vectorized=False,
        )

        if not sol.success:
            # Fallbacks: try alternative stiff/non-stiff solvers
            for alt in ("BDF", "LSODA", "DOP853"):
                sol = solve_ivp(
                    f_ivp,
                    (t0, t1),
                    y0,
                    method=alt,
                    jac=j_ivp if alt in ("BDF", "Radau") else None,
                    rtol=rtol,
                    atol=atol,
                    t_eval=(t1,),
                    dense_output=False,
                    vectorized=False,
                )
                if sol.success:
                    break

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        y_final = sol.y[:, -1]
        if not np.all(np.isfinite(y_final)):
            raise RuntimeError("Non-finite values in solution.")
        return y_final.tolist()