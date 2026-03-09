from typing import Any

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, inline="always")
def _brusselator_rhs(x: float, y: float, A: float, B: float) -> tuple[float, float]:
    xy = x * x * y
    dx = A + xy - (B + 1.0) * x
    dy = B * x - xy
    return dx, dy

@njit(cache=True, fastmath=True, inline="always")
def _rms_norm(v0: float, v1: float) -> float:
    return np.sqrt(0.5 * (v0 * v0 + v1 * v1))

@njit(cache=True, fastmath=True, inline="always")
def _rms_norm_sq(v0: float, v1: float) -> float:
    return 0.5 * (v0 * v0 + v1 * v1)

@njit(cache=True, fastmath=True)
def _select_initial_step(
    t0: float,
    t1: float,
    x0: float,
    y0: float,
    f0x: float,
    f0y: float,
    A: float,
    B: float,
    rtol: float,
    atol: float,
) -> float:
    interval = abs(t1 - t0)
    if interval == 0.0:
        return 0.0

    direction = 1.0
    if t1 < t0:
        direction = -1.0

    scale_x = atol + abs(x0) * rtol
    scale_y = atol + abs(y0) * rtol

    d0 = _rms_norm(x0 / scale_x, y0 / scale_y)
    d1 = _rms_norm(f0x / scale_x, f0y / scale_y)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    if h0 > interval:
        h0 = interval

    x1 = x0 + direction * h0 * f0x
    y1 = y0 + direction * h0 * f0y
    f1x, f1y = _brusselator_rhs(x1, y1, A, B)

    d2 = _rms_norm((f1x - f0x) / scale_x, (f1y - f0y) / scale_y) / max(h0, 1e-16)

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2

    h = min(interval, min(100.0 * h0, h1))
    if h <= 1e-16:
        h = min(interval, 1e-6)
    return h

@njit(cache=True, fastmath=True)
def _solve_brusselator_numba(
    t0: float,
    t1: float,
    x: float,
    y: float,
    A: float,
    B: float,
    rtol: float,
    atol: float,
) -> np.ndarray:
    if t0 == t1:
        return np.array((x, y), dtype=np.float64)

    # Dormand-Prince RK45 coefficients.
    a21 = 1.0 / 5.0

    a31 = 3.0 / 40.0
    a32 = 9.0 / 40.0

    a41 = 44.0 / 45.0
    a42 = -56.0 / 15.0
    a43 = 32.0 / 9.0

    a51 = 19372.0 / 6561.0
    a52 = -25360.0 / 2187.0
    a53 = 64448.0 / 6561.0
    a54 = -212.0 / 729.0

    a61 = 9017.0 / 3168.0
    a62 = -355.0 / 33.0
    a63 = 46732.0 / 5247.0
    a64 = 49.0 / 176.0
    a65 = -5103.0 / 18656.0

    a71 = 35.0 / 384.0
    a73 = 500.0 / 1113.0
    a74 = 125.0 / 192.0
    a75 = -2187.0 / 6784.0
    a76 = 11.0 / 84.0

    e1 = -71.0 / 57600.0
    e3 = 71.0 / 16695.0
    e4 = -71.0 / 1920.0
    e5 = 17253.0 / 339200.0
    e6 = -22.0 / 525.0
    e7 = 1.0 / 40.0

    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    error_exponent = -0.2
    eps64 = 2.220446049250313e-16

    direction = 1.0
    if t1 < t0:
        direction = -1.0

    fx, fy = _brusselator_rhs(x, y, A, B)
    h_abs = _select_initial_step(t0, t1, x, y, fx, fy, A, B, rtol, atol)
    t = t0
    rejected = False

    for _ in range(1000000):
        remaining = direction * (t1 - t)
        if remaining <= 0.0:
            break

        if h_abs > remaining:
            h_abs = remaining

        min_step = 16.0 * eps64 * max(abs(t), abs(t1), 1.0)
        if h_abs < min_step:
            h_abs = min_step
            if h_abs > remaining:
                h_abs = remaining

        h = direction * h_abs

        k1x = fx
        k1y = fy

        sx = x + h * (a21 * k1x)
        sy = y + h * (a21 * k1y)
        k2x, k2y = _brusselator_rhs(sx, sy, A, B)

        sx = x + h * (a31 * k1x + a32 * k2x)
        sy = y + h * (a31 * k1y + a32 * k2y)
        k3x, k3y = _brusselator_rhs(sx, sy, A, B)

        sx = x + h * (a41 * k1x + a42 * k2x + a43 * k3x)
        sy = y + h * (a41 * k1y + a42 * k2y + a43 * k3y)
        k4x, k4y = _brusselator_rhs(sx, sy, A, B)

        sx = x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x)
        sy = y + h * (a51 * k1y + a52 * k2y + a53 * k3y + a54 * k4y)
        k5x, k5y = _brusselator_rhs(sx, sy, A, B)

        sx = x + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x)
        sy = y + h * (a61 * k1y + a62 * k2y + a63 * k3y + a64 * k4y + a65 * k5y)
        k6x, k6y = _brusselator_rhs(sx, sy, A, B)

        x_new = x + h * (a71 * k1x + a73 * k3x + a74 * k4x + a75 * k5x + a76 * k6x)
        y_new = y + h * (a71 * k1y + a73 * k3y + a74 * k4y + a75 * k5y + a76 * k6y)

        k7x, k7y = _brusselator_rhs(x_new, y_new, A, B)

        errx = h * (e1 * k1x + e3 * k3x + e4 * k4x + e5 * k5x + e6 * k6x + e7 * k7x)
        erry = h * (e1 * k1y + e3 * k3y + e4 * k4y + e5 * k5y + e6 * k6y + e7 * k7y)

        scale_x = atol + rtol * max(abs(x), abs(x_new))
        scale_y = atol + rtol * max(abs(y), abs(y_new))
        err_norm_sq = _rms_norm_sq(errx / scale_x, erry / scale_y)

        if err_norm_sq <= 1.0:
            t = t + h
            x = x_new
            y = y_new
            fx = k7x
            fy = k7y

            if err_norm_sq == 0.0:
                factor = max_factor
            else:
                factor = min(max_factor, safety * err_norm_sq ** (-0.1))

            if rejected and factor > 1.0:
                factor = 1.0

            h_abs = h_abs * max(min_factor, factor)
            rejected = False
        else:
            factor = max(min_factor, safety * err_norm_sq ** (-0.1))
            h_abs = h_abs * factor
            rejected = True
            rejected = True

    return np.array((x, y), dtype=np.float64)

class Solver:
    def __init__(self) -> None:
        # Trigger numba compilation ahead of time; init cost does not count.
        _solve_brusselator_numba(0.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1e-8, 1e-8)

    def solve(self, problem, **kwargs) -> Any:
        y0 = problem["y0"]
        params = problem["params"]
        return _solve_brusselator_numba(
            float(problem["t0"]),
            float(problem["t1"]),
            float(y0[0]),
            float(y0[1]),
            float(params["A"]),
            float(params["B"]),
            1e-8,
            1e-8,
        )