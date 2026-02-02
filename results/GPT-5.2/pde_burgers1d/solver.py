from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

if njit is not None:

    @njit(cache=True)
    def _rhs_burgers(u: np.ndarray, nu: float, inv_dx: float, inv_dx2: float, du: np.ndarray) -> None:
        n = u.shape[0]
        for i in range(n):
            ui = u[i]
            uim1 = 0.0 if i == 0 else u[i - 1]
            uip1 = 0.0 if i == n - 1 else u[i + 1]

            # Diffusion: central second derivative
            diffusion = (uip1 - 2.0 * ui + uim1) * inv_dx2

            # Advection: upwind first derivative
            if ui >= 0.0:
                dudx = (ui - uim1) * inv_dx
            else:
                dudx = (uip1 - ui) * inv_dx

            du[i] = -(ui * dudx) + nu * diffusion

    @njit(cache=True)
    def _rhs_burgers_ret(u: np.ndarray, nu: float, inv_dx: float, inv_dx2: float) -> np.ndarray:
        du = np.empty_like(u)
        _rhs_burgers(u, nu, inv_dx, inv_dx2, du)
        return du

    @njit(cache=True)
    def _rk45_step(
        u: np.ndarray,
        nu: float,
        inv_dx: float,
        inv_dx2: float,
        h: float,
        k1: np.ndarray,
        k2: np.ndarray,
        k3: np.ndarray,
        k4: np.ndarray,
        k5: np.ndarray,
        k6: np.ndarray,
        k7: np.ndarray,
        tmp: np.ndarray,
        u5: np.ndarray,
        err: np.ndarray,
    ) -> None:
        # Dormand-Prince coefficients (same family as SciPy RK45)
        # Stage 1
        _rhs_burgers(u, nu, inv_dx, inv_dx2, k1)

        # Stage 2
        n = u.shape[0]
        a21 = 1.0 / 5.0
        for i in range(n):
            tmp[i] = u[i] + h * a21 * k1[i]
        _rhs_burgers(tmp, nu, inv_dx, inv_dx2, k2)

        # Stage 3
        a31 = 3.0 / 40.0
        a32 = 9.0 / 40.0
        for i in range(n):
            tmp[i] = u[i] + h * (a31 * k1[i] + a32 * k2[i])
        _rhs_burgers(tmp, nu, inv_dx, inv_dx2, k3)

        # Stage 4
        a41 = 44.0 / 45.0
        a42 = -56.0 / 15.0
        a43 = 32.0 / 9.0
        for i in range(n):
            tmp[i] = u[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
        _rhs_burgers(tmp, nu, inv_dx, inv_dx2, k4)

        # Stage 5
        a51 = 19372.0 / 6561.0
        a52 = -25360.0 / 2187.0
        a53 = 64448.0 / 6561.0
        a54 = -212.0 / 729.0
        for i in range(n):
            tmp[i] = u[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
        _rhs_burgers(tmp, nu, inv_dx, inv_dx2, k5)

        # Stage 6
        a61 = 9017.0 / 3168.0
        a62 = -355.0 / 33.0
        a63 = 46732.0 / 5247.0
        a64 = 49.0 / 176.0
        a65 = -5103.0 / 18656.0
        for i in range(n):
            tmp[i] = u[i] + h * (
                a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]
            )
        _rhs_burgers(tmp, nu, inv_dx, inv_dx2, k6)

        # 5th order solution (u5)
        b1 = 35.0 / 384.0
        b3 = 500.0 / 1113.0
        b4 = 125.0 / 192.0
        b5 = -2187.0 / 6784.0
        b6 = 11.0 / 84.0
        for i in range(n):
            u5[i] = u[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])

        # Stage 7 (for error estimate)
        _rhs_burgers(u5, nu, inv_dx, inv_dx2, k7)

        # 4th order solution (u4) and error = u5 - u4
        bh1 = 5179.0 / 57600.0
        bh3 = 7571.0 / 16695.0
        bh4 = 393.0 / 640.0
        bh5 = -92097.0 / 339200.0
        bh6 = 187.0 / 2100.0
        bh7 = 1.0 / 40.0
        for i in range(n):
            u4 = u[i] + h * (
                bh1 * k1[i] + bh3 * k3[i] + bh4 * k4[i] + bh5 * k5[i] + bh6 * k6[i] + bh7 * k7[i]
            )
            err[i] = u5[i] - u4

    @njit(cache=True)
    def _integrate_rk45_adaptive(
        u0: np.ndarray,
        t0: float,
        t1: float,
        nu: float,
        dx: float,
        rtol: float,
        atol: float,
    ) -> np.ndarray:
        u = u0.copy()
        n = u.shape[0]
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx

        # Work arrays
        k1 = np.empty(n, dtype=np.float64)
        k2 = np.empty(n, dtype=np.float64)
        k3 = np.empty(n, dtype=np.float64)
        k4 = np.empty(n, dtype=np.float64)
        k5 = np.empty(n, dtype=np.float64)
        k6 = np.empty(n, dtype=np.float64)
        k7 = np.empty(n, dtype=np.float64)
        tmp = np.empty(n, dtype=np.float64)
        u5 = np.empty(n, dtype=np.float64)
        err = np.empty(n, dtype=np.float64)

        t = t0
        # Initial step guess (similar scale to RK45 defaults, but we also cap for stability).
        h = min(1e-3, t1 - t0)
        if h <= 0.0:
            return u

        # Safety / bounds
        safety = 0.9
        min_factor = 0.2
        max_factor = 5.0
        power = -0.2  # -1/(order+1) with order=4 error -> exponent -1/5

        # Avoid infinite loops
        h_min = 1e-12

        while t < t1:
            if t + h > t1:
                h = t1 - t

            # Mild stability caps (advection + diffusion)
            umax = 0.0
            for i in range(n):
                ai = abs(u[i])
                if ai > umax:
                    umax = ai
            h_cap_adv = 0.45 * dx / (umax + 1e-12)
            h_cap_diff = 0.25 * dx * dx / (nu + 1e-18)
            h_cap = h_cap_adv if h_cap_adv < h_cap_diff else h_cap_diff
            if h > h_cap:
                h = h_cap
                if t + h > t1:
                    h = t1 - t

            _rk45_step(u, nu, inv_dx, inv_dx2, h, k1, k2, k3, k4, k5, k6, k7, tmp, u5, err)

            # Error norm (RMS)
            err_sum = 0.0
            for i in range(n):
                scale = atol + rtol * (abs(u[i]) if abs(u[i]) > abs(u5[i]) else abs(u5[i]))
                e = err[i] / scale
                err_sum += e * e
            err_norm = (err_sum / n) ** 0.5

            if err_norm <= 1.0 or h <= h_min:
                # accept
                t += h
                for i in range(n):
                    u[i] = u5[i]

                # step update
                if err_norm == 0.0:
                    factor = max_factor
                else:
                    factor = safety * (err_norm**power)
                    if factor < min_factor:
                        factor = min_factor
                    elif factor > max_factor:
                        factor = max_factor
                h *= factor
            else:
                # reject; shrink
                factor = safety * (err_norm**power)
                if factor < 0.1:
                    factor = 0.1
                h *= factor
                if h < h_min:
                    h = h_min

        return u
class Solver:
    def __init__(self) -> None:
        # Pre-JIT compilation (init not timed).
        if njit is not None:
            u0 = np.zeros(32, dtype=np.float64)
            du0 = np.empty_like(u0)
            _rhs_burgers(u0, 0.005, 10.0, 100.0, du0)

            # Also compile RHS-returning helper used by solve_ivp.
            _ = _rhs_burgers_ret(u0, 0.005, 10.0, 100.0)

    def solve(self, problem: dict, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        nu = float(params["nu"])
        dx = float(params["dx"])

        rtol = 1e-6
        atol = 1e-6
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx

        if njit is not None:

            def fun(_t: float, u: np.ndarray) -> np.ndarray:
                return _rhs_burgers_ret(u, nu, inv_dx, inv_dx2)

        else:

            def fun(_t: float, u: np.ndarray) -> np.ndarray:
                u_pad = np.pad(u, 1, mode="constant", constant_values=0.0)
                diffusion = (u_pad[2:] - 2.0 * u_pad[1:-1] + u_pad[:-2]) * inv_dx2
                du_f = (u_pad[2:] - u_pad[1:-1]) * inv_dx
                du_b = (u_pad[1:-1] - u_pad[:-2]) * inv_dx
                adv = np.where(u >= 0.0, u * du_b, u * du_f)
                return -adv + nu * diffusion

        # Lower overhead than solve_ivp, same underlying RK45 algorithm.
        from scipy.integrate import RK45  # type: ignore

        solver = RK45(fun, t0, y0, t1, rtol=rtol, atol=atol)
        while solver.status == "running":
            solver.step()

        if solver.status != "finished":
            raise RuntimeError("RK45 failed")
        return solver.y.tolist()