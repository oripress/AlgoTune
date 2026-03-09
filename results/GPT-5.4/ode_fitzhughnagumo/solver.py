from __future__ import annotations

from typing import Any
import math

from numba import njit
import numpy as np
from scipy.integrate import RK45

@njit(cache=True)
def _rk45_numba(
    t0: float,
    t1: float,
    v0: float,
    w0: float,
    a: float,
    b: float,
    c: float,
    I: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, int]:
    dt = t1 - t0
    if dt == 0.0:
        return v0, w0, 1

    direction = 1.0 if dt > 0.0 else -1.0
    interval = abs(dt)

    f0v = v0 - (v0 * v0 * v0) / 3.0 - w0 + I
    f0w = a * (b * v0 - c * w0)

    scale_v = atol + abs(v0) * rtol
    scale_w = atol + abs(w0) * rtol
    d0 = math.sqrt(0.5 * ((v0 / scale_v) ** 2 + (w0 / scale_w) ** 2))
    d1 = math.sqrt(0.5 * ((f0v / scale_v) ** 2 + (f0w / scale_w) ** 2))

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    if h0 > interval:
        h0 = interval

    v1 = v0 + direction * h0 * f0v
    w1 = w0 + direction * h0 * f0w
    f1v = v1 - (v1 * v1 * v1) / 3.0 - w1 + I
    f1w = a * (b * v1 - c * w1)

    d2 = math.sqrt(
        0.5 * (((f1v - f0v) / scale_v) ** 2 + ((f1w - f0w) / scale_w) ** 2)
    ) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2

    h = direction * min(100.0 * h0, h1, interval)

    t = t0
    v = v0
    w = w0

    k1v = f0v
    k1w = f0w
    have_k1 = True

    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    rejected = False

    for _ in range(2_000_000):
        remaining = t1 - t
        if abs(remaining) <= 1e-14 * (1.0 + abs(t1)):
            return v, w, 1
        if direction * h > direction * remaining:
            h = remaining

        if not have_k1:
            k1v = v - (v * v * v) / 3.0 - w + I
            k1w = a * (b * v - c * w)

        vv = v + h * (1.0 / 5.0) * k1v
        ww = w + h * (1.0 / 5.0) * k1w
        k2v = vv - (vv * vv * vv) / 3.0 - ww + I
        k2w = a * (b * vv - c * ww)

        vv = v + h * ((3.0 / 40.0) * k1v + (9.0 / 40.0) * k2v)
        ww = w + h * ((3.0 / 40.0) * k1w + (9.0 / 40.0) * k2w)
        k3v = vv - (vv * vv * vv) / 3.0 - ww + I
        k3w = a * (b * vv - c * ww)

        vv = v + h * ((44.0 / 45.0) * k1v + (-56.0 / 15.0) * k2v + (32.0 / 9.0) * k3v)
        ww = w + h * ((44.0 / 45.0) * k1w + (-56.0 / 15.0) * k2w + (32.0 / 9.0) * k3w)
        k4v = vv - (vv * vv * vv) / 3.0 - ww + I
        k4w = a * (b * vv - c * ww)

        vv = v + h * (
            (19372.0 / 6561.0) * k1v
            + (-25360.0 / 2187.0) * k2v
            + (64448.0 / 6561.0) * k3v
            + (-212.0 / 729.0) * k4v
        )
        ww = w + h * (
            (19372.0 / 6561.0) * k1w
            + (-25360.0 / 2187.0) * k2w
            + (64448.0 / 6561.0) * k3w
            + (-212.0 / 729.0) * k4w
        )
        k5v = vv - (vv * vv * vv) / 3.0 - ww + I
        k5w = a * (b * vv - c * ww)

        vv = v + h * (
            (9017.0 / 3168.0) * k1v
            + (-355.0 / 33.0) * k2v
            + (46732.0 / 5247.0) * k3v
            + (49.0 / 176.0) * k4v
            + (-5103.0 / 18656.0) * k5v
        )
        ww = w + h * (
            (9017.0 / 3168.0) * k1w
            + (-355.0 / 33.0) * k2w
            + (46732.0 / 5247.0) * k3w
            + (49.0 / 176.0) * k4w
            + (-5103.0 / 18656.0) * k5w
        )
        k6v = vv - (vv * vv * vv) / 3.0 - ww + I
        k6w = a * (b * vv - c * ww)

        v_new = v + h * (
            (35.0 / 384.0) * k1v
            + (500.0 / 1113.0) * k3v
            + (125.0 / 192.0) * k4v
            + (-2187.0 / 6784.0) * k5v
            + (11.0 / 84.0) * k6v
        )
        w_new = w + h * (
            (35.0 / 384.0) * k1w
            + (500.0 / 1113.0) * k3w
            + (125.0 / 192.0) * k4w
            + (-2187.0 / 6784.0) * k5w
            + (11.0 / 84.0) * k6w
        )

        k7v = v_new - (v_new * v_new * v_new) / 3.0 - w_new + I
        k7w = a * (b * v_new - c * w_new)

        ev = h * (
            (71.0 / 57600.0) * k1v
            + (-71.0 / 16695.0) * k3v
            + (71.0 / 1920.0) * k4v
            + (-17253.0 / 339200.0) * k5v
            + (22.0 / 525.0) * k6v
            + (-1.0 / 40.0) * k7v
        )
        ew = h * (
            (71.0 / 57600.0) * k1w
            + (-71.0 / 16695.0) * k3w
            + (71.0 / 1920.0) * k4w
            + (-17253.0 / 339200.0) * k5w
            + (22.0 / 525.0) * k6w
            + (-1.0 / 40.0) * k7w
        )

        scale_v = atol + max(abs(v), abs(v_new)) * rtol
        scale_w = atol + max(abs(w), abs(w_new)) * rtol
        err_norm = math.sqrt(0.5 * ((ev / scale_v) ** 2 + (ew / scale_w) ** 2))

        if err_norm < 1.0:
            t += h
            v = v_new
            w = w_new
            k1v = k7v
            k1w = k7w
            have_k1 = True

            if err_norm == 0.0:
                factor = max_factor
            else:
                factor = safety * (err_norm ** -0.2)
                if rejected and factor > 1.0:
                    factor = 1.0
                if factor < min_factor:
                    factor = min_factor
                elif factor > max_factor:
                    factor = max_factor
            h *= factor
            rejected = False
        else:
            factor = safety * (err_norm ** -0.2)
            if factor < min_factor:
                factor = min_factor
            elif factor > 1.0:
                factor = 1.0
            h *= factor
            rejected = True
            have_k1 = True
            if abs(h) <= 1e-15 * (1.0 + abs(t)):
                return v, w, 0

    return v, w, 0

class Solver:
    def __init__(self) -> None:
        _rk45_numba(0.0, 1.0, -1.0, -0.5, 0.08, 0.8, 0.7, 0.5, 1e-8, 1e-8)

    def solve(self, problem, **kwargs) -> Any:
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]

        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])
        v0 = float(y0[0])
        w0 = float(y0[1])

        v1, w1, ok1 = _rk45_numba(t0, t1, v0, w0, a, b, c, I, 1e-8, 1e-8)
        if ok1:
            v2, w2, ok2 = _rk45_numba(t0, t1, v0, w0, a, b, c, I, 3e-9, 3e-9)
            if ok2:
                tv = 1e-8 + 5e-7 * max(abs(v1), abs(v2))
                tw = 1e-8 + 5e-7 * max(abs(w1), abs(w2))
                if abs(v1 - v2) <= tv and abs(w1 - w2) <= tw:
                    return [float(v2), float(w2)]

        def fun(_t: float, y: np.ndarray) -> np.ndarray:
            v = y[0]
            w = y[1]
            return np.array(
                [
                    v - (v * v * v) / 3.0 - w + I,
                    a * (b * v - c * w),
                ],
                dtype=float,
            )

        solver = RK45(fun, t0, np.array([v0, w0]), t1, rtol=1e-8, atol=1e-8, vectorized=False)
        while solver.status == "running":
            solver.step()
        return [float(solver.y[0]), float(solver.y[1])]