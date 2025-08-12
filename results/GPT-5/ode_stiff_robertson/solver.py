from __future__ import annotations

from typing import Any
import numpy as np
from scipy.integrate import odeint, ode, solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        # Parse inputs
        y0 = np.asarray(problem["y0"], dtype=float).reshape(3)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k1, k2, k3 = map(float, problem["k"])

        if t1 == t0:
            return y0.tolist()
        if t1 < t0:
            # Not expected; return initial to avoid undefined behavior
            return y0.tolist()

        # Use mass conservation to reduce dimension: y3 = S - y1 - y2
        S = float(y0.sum())
        z0 = y0[:2].copy()  # [y1, y2]

        # RHS and Jacobian for 2D reduced system (odeint expects y first, then t)
        def f2(z, t, k1=k1, k2=k2, k3=k3, S=S):
            y1 = z[0]
            y2 = z[1]
            y3 = S - y1 - y2
            y2y3 = y2 * y3
            return (-k1 * y1 + k3 * y2y3, k1 * y1 - k2 * y2 * y2 - k3 * y2y3)

        def j2(z, t, k1=k1, k2=k2, k3=k3, S=S):
            y1 = z[0]
            y2 = z[1]
            temp = S - y1 - 2.0 * y2
            # Return as nested lists to avoid NumPy array construction overhead
            return [
                [-k1 - k3 * y2, k3 * temp],
                [k1 + k3 * y2, -2.0 * k2 * y2 - k3 * temp],
            ]

        # Primary: low-overhead ODEPACK LSODA via odeint with analytic Jacobian
        try:
            tspan = [t0, t1]
            z = odeint(
                f2,
                z0,
                tspan,
                Dfun=j2,
                col_deriv=0,
                rtol=5e-6,
                atol=1e-9,
                mxstep=100000,
                # hmax left as default (0.0) for no artificial limit
            )
            y1f, y2f = float(z[-1, 0]), float(z[-1, 1])
        except Exception:
            # Fallback 1: VODE (BDF) with analytic Jacobian
            try:
                def f2_ode(t: float, z, k1=k1, k2=k2, k3=k3, S=S):
                    y1 = z[0]
                    y2 = z[1]
                    y3 = S - y1 - y2
                    y2y3 = y2 * y3
                    return (-k1 * y1 + k3 * y2y3, k1 * y1 - k2 * y2 * y2 - k3 * y2y3)

                def j2_ode(t: float, z, k1=k1, k2=k2, k3=k3, S=S):
                    y1 = z[0]
                    y2 = z[1]
                    temp = S - y1 - 2.0 * y2
                    return (
                        (-k1 - k3 * y2, k3 * temp),
                        (k1 + k3 * y2, -2.0 * k2 * y2 - k3 * temp),
                    )

                r = ode(f2_ode, j2_ode)
                r.set_integrator(
                    "vode",
                    method="bdf",
                    with_jacobian=True,
                    rtol=2e-6,
                    atol=1e-9,
                    nsteps=100000,
                    max_step=(t1 - t0),
                )
                r.set_initial_value(z0, t0)
                zf = r.integrate(t1)
                if not r.successful():
                    raise RuntimeError("VODE-BDF unsuccessful")
                y1f, y2f = zf[0], zf[1]
            except Exception:
                # Fallback 2: LSODA via solve_ivp
                def f2_ivp(_t: float, z, k1=k1, k2=k2, k3=k3, S=S):
                    y1 = z[0]
                    y2 = z[1]
                    y3 = S - y1 - y2
                    y2y3 = y2 * y3
                    return (-k1 * y1 + k3 * y2y3, k1 * y1 - k2 * y2 * y2 - k3 * y2y3)

                sol = solve_ivp(
                    f2_ivp,
                    (t0, t1),
                    z0,
                    method="LSODA",
                    rtol=2e-6,
                    atol=1e-9,
                    dense_output=False,
                    vectorized=False,
                )
                if not sol.success:
                    raise RuntimeError(f"Solve failed: {sol.message}")
                y1f, y2f = sol.y[:, -1]

        y3f = S - y1f - y2f
        return [float(y1f), float(y2f), float(y3f)]