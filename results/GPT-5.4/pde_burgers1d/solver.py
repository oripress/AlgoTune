from __future__ import annotations

from typing import Any

import math
import numpy as np
from numba import njit
from scipy.integrate import RK45

@njit(cache=True, fastmath=True)
def _rhs(u: np.ndarray, out: np.ndarray, nu: float, inv_dx: float, inv_dx2: float) -> None:
    n = u.shape[0]
    if n == 0:
        return

    ui = u[0]
    u_right = u[1] if n > 1 else 0.0
    if ui >= 0.0:
        dudx = ui * inv_dx
    else:
        dudx = (u_right - ui) * inv_dx
    diffusion = (u_right - 2.0 * ui) * inv_dx2
    out[0] = -ui * dudx + nu * diffusion

    for i in range(1, n - 1):
        u_left = u[i - 1]
        ui = u[i]
        u_right = u[i + 1]
        if ui >= 0.0:
            dudx = (ui - u_left) * inv_dx
        else:
            dudx = (u_right - ui) * inv_dx
        diffusion = (u_right - 2.0 * ui + u_left) * inv_dx2
        out[i] = -ui * dudx + nu * diffusion

    if n > 1:
        ui = u[n - 1]
        u_left = u[n - 2]
        if ui >= 0.0:
            dudx = (ui - u_left) * inv_dx
        else:
            dudx = -ui * inv_dx
        diffusion = (-2.0 * ui + u_left) * inv_dx2
        out[n - 1] = -ui * dudx + nu * diffusion

@njit(cache=True)
def _dopri5_integrate(
    y0: np.ndarray,
    total_time: float,
    nu: float,
    dx: float,
    max_step: float,
    rtol: float,
    atol: float,
) -> np.ndarray:
    y = y0.copy()
    n = y.shape[0]
    if n == 0 or total_time <= 0.0:
        return y

    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx

    k1 = np.empty(n, dtype=np.float64)
    k2 = np.empty(n, dtype=np.float64)
    k3 = np.empty(n, dtype=np.float64)
    k4 = np.empty(n, dtype=np.float64)
    k5 = np.empty(n, dtype=np.float64)
    k6 = np.empty(n, dtype=np.float64)
    k7 = np.empty(n, dtype=np.float64)
    temp = np.empty(n, dtype=np.float64)
    y_new = np.empty(n, dtype=np.float64)

    _rhs(y, k1, nu, inv_dx, inv_dx2)
    h_abs = max_step
    if h_abs > 0.05:
        h_abs = 0.05
    if h_abs > total_time:
        h_abs = total_time

    t = 0.0
    step_rejected = False
    min_step = 1e-15
    max_iters = 1000000
    iters = 0

    while t < total_time and iters < max_iters:
        iters += 1
        h = h_abs
        remaining = total_time - t
        if h > remaining:
            h = remaining

        for i in range(n):
            temp[i] = y[i] + h * (1.0 / 5.0) * k1[i]
        _rhs(temp, k2, nu, inv_dx, inv_dx2)

        for i in range(n):
            temp[i] = y[i] + h * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i])
        _rhs(temp, k3, nu, inv_dx, inv_dx2)

        for i in range(n):
            temp[i] = y[i] + h * (
                (44.0 / 45.0) * k1[i]
                + (-56.0 / 15.0) * k2[i]
                + (32.0 / 9.0) * k3[i]
            )
        _rhs(temp, k4, nu, inv_dx, inv_dx2)

        for i in range(n):
            temp[i] = y[i] + h * (
                (19372.0 / 6561.0) * k1[i]
                + (-25360.0 / 2187.0) * k2[i]
                + (64448.0 / 6561.0) * k3[i]
                + (-212.0 / 729.0) * k4[i]
            )
        _rhs(temp, k5, nu, inv_dx, inv_dx2)

        for i in range(n):
            temp[i] = y[i] + h * (
                (9017.0 / 3168.0) * k1[i]
                + (-355.0 / 33.0) * k2[i]
                + (46732.0 / 5247.0) * k3[i]
                + (49.0 / 176.0) * k4[i]
                + (-5103.0 / 18656.0) * k5[i]
            )
        _rhs(temp, k6, nu, inv_dx, inv_dx2)

        for i in range(n):
            y_new[i] = y[i] + h * (
                (35.0 / 384.0) * k1[i]
                + (500.0 / 1113.0) * k3[i]
                + (125.0 / 192.0) * k4[i]
                + (-2187.0 / 6784.0) * k5[i]
                + (11.0 / 84.0) * k6[i]
            )
        _rhs(y_new, k7, nu, inv_dx, inv_dx2)

        err_sq = 0.0
        for i in range(n):
            err = h * (
                (-71.0 / 57600.0) * k1[i]
                + (71.0 / 16695.0) * k3[i]
                + (-71.0 / 1920.0) * k4[i]
                + (17253.0 / 339200.0) * k5[i]
                + (-22.0 / 525.0) * k6[i]
                + (1.0 / 40.0) * k7[i]
            )
            scale = atol + rtol * max(abs(y[i]), abs(y_new[i]))
            z = err / scale
            err_sq += z * z
        err_norm = math.sqrt(err_sq / n)

        if err_norm <= 1.0:
            t += h
            for i in range(n):
                y[i] = y_new[i]
                k1[i] = k7[i]

            if err_norm == 0.0:
                factor = 10.0
            else:
                factor = 0.9 * (err_norm ** -0.2)
                if factor > 10.0:
                    factor = 10.0
                elif factor < 0.2:
                    factor = 0.2

            if step_rejected and factor > 1.0:
                factor = 1.0

            h_abs = h * factor
            if h_abs > max_step:
                h_abs = max_step
            step_rejected = False
        else:
            factor = 0.9 * (err_norm ** -0.2)
            if factor < 0.2:
                factor = 0.2
            h_abs = h * factor
            step_rejected = True

        if h_abs < min_step:
            h_abs = min_step

    return y
class Solver:
    def __init__(self) -> None:
        dummy = np.zeros(4, dtype=np.float64)
        out = np.zeros(4, dtype=np.float64)
        _rhs(dummy, out, 0.005, 10.0, 100.0)

    def solve(self, problem, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        nu = float(params["nu"])
        dx = float(params["dx"])

        if t1 <= t0 or y0.size == 0:
            return y0.tolist()

        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        out = np.empty_like(y0)

        def burgers_equation(_t: float, u: np.ndarray) -> np.ndarray:
            _rhs(u, out, nu, inv_dx, inv_dx2)
            return out.copy()

        solver = RK45(
            burgers_equation,
            t0,
            y0,
            t1,
            rtol=1e-6,
            atol=1e-6,
        )
        step = solver.step
        while solver.status == "running":
            step()
        if solver.status != "finished":
            raise RuntimeError("Solver failed")
        return solver.y.tolist()