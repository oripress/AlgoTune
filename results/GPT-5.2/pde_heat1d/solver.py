from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import dst, idst

class Solver:
    """
    Fast solver for the semi-discrete 1D heat equation with Dirichlet boundaries.

    Semi-discrete linear ODE:
        du/dt = (alpha/dx^2) * L u
    where L is the tridiagonal second-difference operator with Dirichlet BCs.

    Using DST-I, L is diagonalizable, enabling an exact matrix exponential
    application in O(n log n).
    """

    # Cache exp(eigs*dt) factors for repeated (n, theta) pairs.
    _exp_cache: dict[tuple[int, float], np.ndarray] = {}

    def solve(self, problem: dict, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        dt = t1 - t0
        if dt == 0.0:
            return y0.tolist()

        params = problem["params"]
        alpha = float(params["alpha"])
        dx = float(params["dx"])

        n = int(y0.size)
        theta = alpha * dt / (dx * dx)

        key = (n, theta)
        exp_fac = self._exp_cache.get(key)
        if exp_fac is None:
            k = np.arange(1, n + 1, dtype=np.float64)
            s = np.sin((0.5 * np.pi / (n + 1)) * k)
            # exp( (alpha*dt/dx^2) * (-4*sin^2(...)) )
            exp_fac = np.exp((-4.0 * theta) * (s * s))
            self._exp_cache[key] = exp_fac

        y_hat = dst(y0, type=1, norm="ortho")
        y_hat *= exp_fac
        y1 = np.asarray(idst(y_hat, type=1, norm="ortho"), dtype=np.float64)

        # Optional local debug: compare to solve_ivp (slow; never enabled in eval).
        dbg = problem.get("__debug_compare_solve_ivp", 0)
        if dbg:
            from scipy.integrate import solve_ivp

            def heat_equation(_t: float, u: np.ndarray) -> np.ndarray:
                u_padded = np.pad(u, 1, mode="constant", constant_values=0.0)
                u_xx = (u_padded[2:] - 2.0 * u_padded[1:-1] + u_padded[:-2]) / (dx * dx)
                return alpha * u_xx

            if int(dbg) == 1:
                t_eval = None
                dense_output = False
            else:
                t_eval = np.linspace(t0, t1, 1000)
                dense_output = True

            sol = solve_ivp(
                heat_equation,
                [t0, t1],
                y0,
                method="RK45",
                rtol=1e-6,
                atol=1e-6,
                t_eval=t_eval,
                dense_output=dense_output,
            )
            y_ref = sol.y[:, -1]
            diff = y1 - y_ref
            max_abs = float(np.max(np.abs(diff)))
            max_rel = float(np.max(np.abs(diff) / (1e-12 + np.abs(y_ref))))
            print(
                "debug solve_ivp",
                "mode",
                int(dbg),
                "success",
                bool(sol.success),
                "max_abs",
                max_abs,
                "max_rel",
                max_rel,
            )

        return y1.tolist()