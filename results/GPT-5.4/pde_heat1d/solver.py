from typing import Any

import numpy as np
from numba import njit
from scipy.fft import dst
from scipy.integrate import RK45

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
ERROR_EXPONENT = -0.2
RTOL = 1e-6
ATOL = 1e-6

A21 = 1.0 / 5.0
A31 = 3.0 / 40.0
A32 = 9.0 / 40.0
A41 = 44.0 / 45.0
A42 = -56.0 / 15.0
A43 = 32.0 / 9.0
A51 = 19372.0 / 6561.0
A52 = -25360.0 / 2187.0
A53 = 64448.0 / 6561.0
A54 = -212.0 / 729.0
A61 = 9017.0 / 3168.0
A62 = -355.0 / 33.0
A63 = 46732.0 / 5247.0
A64 = 49.0 / 176.0
A65 = -5103.0 / 18656.0

B1 = 35.0 / 384.0
B3 = 500.0 / 1113.0
B4 = 125.0 / 192.0
B5 = -2187.0 / 6784.0
B6 = 11.0 / 84.0

E1 = -71.0 / 57600.0
E3 = 71.0 / 16695.0
E4 = -71.0 / 1920.0
E5 = 17253.0 / 339200.0
E6 = -22.0 / 525.0
E7 = 1.0 / 40.0

@njit(cache=True)
def _rhs(u: np.ndarray, coef: float, out: np.ndarray) -> None:
    n = u.shape[0]
    if n == 1:
        out[0] = -2.0 * coef * u[0]
        return
    out[0] = coef * (u[1] - 2.0 * u[0])
    for i in range(1, n - 1):
        out[i] = coef * (u[i + 1] - 2.0 * u[i] + u[i - 1])
    out[n - 1] = coef * (u[n - 2] - 2.0 * u[n - 1])

@njit(cache=True)
def _norm_scaled(v: np.ndarray, scale: np.ndarray) -> float:
    s = 0.0
    n = v.shape[0]
    for i in range(n):
        z = v[i] / scale[i]
        s += z * z
    return np.sqrt(s / n)

@njit(cache=True)
def _select_initial_step(
    y0: np.ndarray, t0: float, t1: float, f0: np.ndarray, direction: float, coef: float
) -> float:
    interval_length = abs(t1 - t0)
    if interval_length == 0.0:
        return 0.0

    n = y0.shape[0]
    scale = np.empty(n, dtype=np.float64)
    for i in range(n):
        scale[i] = ATOL + abs(y0[i]) * RTOL

    d0 = _norm_scaled(y0, scale)
    d1 = _norm_scaled(f0, scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    if h0 > interval_length:
        h0 = interval_length

    y1 = np.empty(n, dtype=np.float64)
    for i in range(n):
        y1[i] = y0[i] + direction * h0 * f0[i]

    f1 = np.empty(n, dtype=np.float64)
    _rhs(y1, coef, f1)

    diff = np.empty(n, dtype=np.float64)
    for i in range(n):
        diff[i] = f1[i] - f0[i]
    d2 = _norm_scaled(diff, scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2

    h = min(100.0 * h0, h1)
    if h > interval_length:
        h = interval_length
    return h

@njit(cache=True)
def _integrate_numba(y0: np.ndarray, t0: float, t1: float, coef: float) -> np.ndarray:
    y = y0.copy()
    n = y.shape[0]
    if t0 == t1:
        return y

    direction = 1.0 if t1 > t0 else -1.0
    f = np.empty(n, dtype=np.float64)
    _rhs(y, coef, f)
    h_abs = _select_initial_step(y, t0, t1, f, direction, coef)

    k2 = np.empty(n, dtype=np.float64)
    k3 = np.empty(n, dtype=np.float64)
    k4 = np.empty(n, dtype=np.float64)
    k5 = np.empty(n, dtype=np.float64)
    k6 = np.empty(n, dtype=np.float64)
    k7 = np.empty(n, dtype=np.float64)
    y_tmp = np.empty(n, dtype=np.float64)
    y_new = np.empty(n, dtype=np.float64)

    t = t0
    while direction * (t1 - t) > 0.0:
        min_step = 10.0 * abs(np.nextafter(t, direction * np.inf) - t)
        if h_abs < min_step:
            h_abs = min_step

        step_rejected = False
        while True:
            if h_abs < min_step:
                return y

            h = direction * h_abs
            t_new = t + h
            if direction * (t_new - t1) > 0.0:
                t_new = t1
            h = t_new - t
            h_abs = abs(h)

            for i in range(n):
                y_tmp[i] = y[i] + h * (A21 * f[i])
            _rhs(y_tmp, coef, k2)

            for i in range(n):
                y_tmp[i] = y[i] + h * (A31 * f[i] + A32 * k2[i])
            _rhs(y_tmp, coef, k3)

            for i in range(n):
                y_tmp[i] = y[i] + h * (A41 * f[i] + A42 * k2[i] + A43 * k3[i])
            _rhs(y_tmp, coef, k4)

            for i in range(n):
                y_tmp[i] = y[i] + h * (A51 * f[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i])
            _rhs(y_tmp, coef, k5)

            for i in range(n):
                y_tmp[i] = y[i] + h * (
                    A61 * f[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]
                )
            _rhs(y_tmp, coef, k6)

            for i in range(n):
                y_new[i] = y[i] + h * (B1 * f[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i])
            _rhs(y_new, coef, k7)

            err_sum = 0.0
            for i in range(n):
                scale = ATOL + max(abs(y[i]), abs(y_new[i])) * RTOL
                err = h * (E1 * f[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i])
                r = err / scale
                err_sum += r * r
            error_norm = np.sqrt(err_sum / n)

            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * (error_norm ** ERROR_EXPONENT))
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                break

            h_abs *= max(MIN_FACTOR, SAFETY * (error_norm ** ERROR_EXPONENT))
            step_rejected = True

        t = t_new
        y, y_new = y_new, y
        f, k7 = k7, f

    return y

class Solver:
    def __init__(self) -> None:
        self._decay_cache: dict[tuple[int, float], np.ndarray] = {}
        dummy = np.array([0.0, 1.0], dtype=np.float64)
        _integrate_numba(dummy, 0.0, 0.1, 1.0)

    def _solve_exact(self, y0: np.ndarray, decay_strength: float) -> np.ndarray:
        n = int(y0.size)
        key = (n, decay_strength)
        decay = self._decay_cache.get(key)
        if decay is None:
            k = np.arange(1, n + 1, dtype=np.float64)
            eig = -4.0 * np.sin(0.5 * np.pi * k / (n + 1.0)) ** 2
            decay = np.exp(decay_strength * eig)
            self._decay_cache[key] = decay
        coeffs = dst(y0, type=1, norm="ortho")
        return np.asarray(dst(coeffs * decay, type=1, norm="ortho"), dtype=np.float64)

    def _solve_rk45(self, y0: np.ndarray, t0: float, t1: float, coef: float) -> list[float]:
        n = int(y0.size)
        if n == 1:
            def fun(t, u):
                return np.array((-2.0 * coef * u[0],), dtype=np.float64)
        else:
            def fun(t, u):
                du = np.empty_like(u)
                du[0] = coef * (u[1] - 2.0 * u[0])
                du[1:-1] = coef * (u[2:] - 2.0 * u[1:-1] + u[:-2])
                du[-1] = coef * (u[-2] - 2.0 * u[-1])
                return du
        solver = RK45(fun, t0, y0, t1, rtol=1e-6, atol=1e-6)
        while solver.status == "running":
            solver.step()
        return list(np.asarray(solver.y, dtype=np.float64))

    def solve(self, problem, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        if y0.size == 0:
            return []

        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        if t0 == t1:
            return list(np.asarray(y0, dtype=np.float64))

        coef = float(problem["params"]["alpha"]) / (float(problem["params"]["dx"]) ** 2)

        y_numba = _integrate_numba(y0, t0, t1, coef)
        y_exact = self._solve_exact(y0, coef * (t1 - t0))

        if np.allclose(y_numba, y_exact, rtol=3e-6, atol=3e-9):
            return list(np.asarray(y_numba, dtype=np.float64))

        return self._solve_rk45(y0, t0, t1, coef)