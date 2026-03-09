from typing import Any
import math

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, inline="always")
def _compute_derivative(y, masses, eps2, n, out):
    d3 = 3 * n
    for i in range(d3):
        out[i] = y[d3 + i]
        out[d3 + i] = 0.0

    for i in range(n - 1):
        ii = 3 * i
        xi = y[ii]
        yi = y[ii + 1]
        zi = y[ii + 2]
        mi = masses[i]

        for j in range(i + 1, n):
            jj = 3 * j
            dx = y[jj] - xi
            dy = y[jj + 1] - yi
            dz = y[jj + 2] - zi

            dist2 = dx * dx + dy * dy + dz * dz + eps2
            invdist = 1.0 / math.sqrt(dist2)
            invdist3 = invdist / dist2

            sij = masses[j] * invdist3
            sji = mi * invdist3

            out[d3 + ii] += sij * dx
            out[d3 + ii + 1] += sij * dy
            out[d3 + ii + 2] += sij * dz

            out[d3 + jj] -= sji * dx
            out[d3 + jj + 1] -= sji * dy
            out[d3 + jj + 2] -= sji * dz

@njit(cache=True, fastmath=True, inline="always")
def _rk45_step(y, f, h, masses, eps2, n, k, y_tmp, y_new, f_new):
    d = y.shape[0]
    for i in range(d):
        k[0, i] = f[i]

    for i in range(d):
        y_tmp[i] = y[i] + h * (0.2 * k[0, i])
    _compute_derivative(y_tmp, masses, eps2, n, k[1])

    for i in range(d):
        y_tmp[i] = y[i] + h * ((3.0 / 40.0) * k[0, i] + (9.0 / 40.0) * k[1, i])
    _compute_derivative(y_tmp, masses, eps2, n, k[2])

    for i in range(d):
        y_tmp[i] = y[i] + h * (
            (44.0 / 45.0) * k[0, i]
            + (-56.0 / 15.0) * k[1, i]
            + (32.0 / 9.0) * k[2, i]
        )
    _compute_derivative(y_tmp, masses, eps2, n, k[3])

    for i in range(d):
        y_tmp[i] = y[i] + h * (
            (19372.0 / 6561.0) * k[0, i]
            + (-25360.0 / 2187.0) * k[1, i]
            + (64448.0 / 6561.0) * k[2, i]
            + (-212.0 / 729.0) * k[3, i]
        )
    _compute_derivative(y_tmp, masses, eps2, n, k[4])

    for i in range(d):
        y_tmp[i] = y[i] + h * (
            (9017.0 / 3168.0) * k[0, i]
            + (-355.0 / 33.0) * k[1, i]
            + (46732.0 / 5247.0) * k[2, i]
            + (49.0 / 176.0) * k[3, i]
            + (-5103.0 / 18656.0) * k[4, i]
        )
    _compute_derivative(y_tmp, masses, eps2, n, k[5])

    for i in range(d):
        y_new[i] = y[i] + h * (
            (35.0 / 384.0) * k[0, i]
            + (500.0 / 1113.0) * k[2, i]
            + (125.0 / 192.0) * k[3, i]
            + (-2187.0 / 6784.0) * k[4, i]
            + (11.0 / 84.0) * k[5, i]
        )

    _compute_derivative(y_new, masses, eps2, n, f_new)

    for i in range(d):
        k[6, i] = f_new[i]

@njit(cache=True, fastmath=True, inline="always")
def _scaled_norm(v, scale):
    s = 0.0
    d = v.shape[0]
    for i in range(d):
        x = v[i] / scale[i]
        s += x * x
    return math.sqrt(s / d)

@njit(cache=True, fastmath=True, inline="always")
def _select_initial_step(y0, f0, t0, t1, masses, eps2, n, rtol, atol, y1, f1, scale):
    d = y0.shape[0]
    interval = abs(t1 - t0)
    if interval == 0.0:
        return 0.0

    for i in range(d):
        scale[i] = atol + abs(y0[i]) * rtol

    d0 = _scaled_norm(y0, scale)
    d1 = _scaled_norm(f0, scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    if h0 > interval:
        h0 = interval

    direction = 1.0 if t1 >= t0 else -1.0
    for i in range(d):
        y1[i] = y0[i] + direction * h0 * f0[i]

    _compute_derivative(y1, masses, eps2, n, f1)

    s = 0.0
    for i in range(d):
        x = (f1[i] - f0[i]) / scale[i]
        s += x * x
    d2 = math.sqrt(s / d) / h0 if h0 > 0.0 else 0.0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2

    h = min(100.0 * h0, h1)
    if h > interval:
        h = interval
    return h

@njit(cache=True, fastmath=True, inline="always")
def _error_norm(y, y_new, h, k, rtol, atol):
    d = y.shape[0]
    s = 0.0

    e0 = -71.0 / 57600.0
    e2 = 71.0 / 16695.0
    e3 = -71.0 / 1920.0
    e4 = 17253.0 / 339200.0
    e5 = -22.0 / 525.0
    e6 = 1.0 / 40.0

    for i in range(d):
        err = h * (
            e0 * k[0, i]
            + e2 * k[2, i]
            + e3 * k[3, i]
            + e4 * k[4, i]
            + e5 * k[5, i]
            + e6 * k[6, i]
        )
        scale = atol + max(abs(y[i]), abs(y_new[i])) * rtol
        x = err / scale
        s += x * x

    return math.sqrt(s / d)

@njit(cache=True, fastmath=True)
def _integrate_endpoint(y0, masses, eps2, t0, t1, rtol, atol):
    y = y0.copy()
    d = y.shape[0]
    n = masses.shape[0]

    if t0 == t1:
        return y

    direction = 1.0 if t1 >= t0 else -1.0
    k = np.empty((7, d), dtype=np.float64)
    y_tmp = np.empty(d, dtype=np.float64)
    y_new = np.empty(d, dtype=np.float64)
    f = np.empty(d, dtype=np.float64)
    f_new = np.empty(d, dtype=np.float64)
    scale = np.empty(d, dtype=np.float64)

    _compute_derivative(y, masses, eps2, n, f)
    h_abs = _select_initial_step(y, f, t0, t1, masses, eps2, n, rtol, atol, y_tmp, f_new, scale)

    if h_abs <= 0.0:
        return y

    t = t0
    step_rejected = False
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    exponent = -0.2

    while direction * (t1 - t) > 0.0:
        remaining = abs(t1 - t)
        if h_abs > remaining:
            h_abs = remaining

        min_step = 1e-15 * max(1.0, abs(t))
        if h_abs < min_step:
            h_abs = min_step
            if h_abs > remaining:
                h_abs = remaining

        h = direction * h_abs
        _rk45_step(y, f, h, masses, eps2, n, k, y_tmp, y_new, f_new)
        err = _error_norm(y, y_new, h, k, rtol, atol)

        if err < 1.0:
            t += h
            for i in range(d):
                y[i] = y_new[i]
                f[i] = f_new[i]

            if err == 0.0:
                factor = max_factor
            else:
                factor = safety * (err ** exponent)
                if factor > max_factor:
                    factor = max_factor

            if step_rejected and factor > 1.0:
                factor = 1.0

            h_abs *= factor
            step_rejected = False
        else:
            factor = safety * (err ** exponent)
            if factor < min_factor:
                factor = min_factor
            h_abs *= factor
            step_rejected = True

    return y

class Solver:
    def __init__(self):
        self._rtol = 1e-8
        self._atol = 1e-8
        dummy_y = np.zeros(12, dtype=np.float64)
        dummy_m = np.array([1.0, 1e-3], dtype=np.float64)
        _integrate_endpoint(dummy_y, dummy_m, 1e-8, 0.0, 1e-3, self._rtol, self._atol)

    def solve(self, problem, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        masses = np.asarray(problem["masses"], dtype=np.float64)
        eps2 = float(problem["softening"]) ** 2
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        return _integrate_endpoint(y0, masses, eps2, t0, t1, self._rtol, self._atol)