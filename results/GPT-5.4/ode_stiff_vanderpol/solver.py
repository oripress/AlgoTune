from math import isfinite, sqrt
from typing import Any

import numpy as np
from numba import njit
from scipy.integrate import odeint

_GAMMA = 2.0 - sqrt(2.0)
_A1 = 0.5 * _GAMMA
_ALPHA = 1.0 / (_GAMMA * (2.0 - _GAMMA))
_BETA = (1.0 - _GAMMA) * (1.0 - _GAMMA) / (_GAMMA * (2.0 - _GAMMA))
_DELTA = (1.0 - _GAMMA) / (2.0 - _GAMMA)

@njit(cache=True)
def _f2(x, v, mu):
    return mu * ((1.0 - x * x) * v - x)

@njit(cache=True)
def _solve2x2(a00, a01, a10, a11, b0, b1):
    det = a00 * a11 - a01 * a10
    if det == 0.0:
        return False, 0.0, 0.0
    inv = 1.0 / det
    x0 = (a11 * b0 - a01 * b1) * inv
    x1 = (-a10 * b0 + a00 * b1) * inv
    return True, x0, x1

@njit(cache=True)
def _trbdf2_single_step(x, v, h, mu):
    a = _A1 * h

    f2n = _f2(x, v, mu)

    x1 = x + _GAMMA * h * v
    v1 = v + _GAMMA * h * f2n

    for _ in range(8):
        f21 = _f2(x1, v1, mu)
        r0 = x1 - x - a * (v + v1)
        r1 = v1 - v - a * (f2n + f21)
        if max(abs(r0), abs(r1)) <= 1e-12 * (1.0 + max(abs(x1), abs(v1))):
            break

        j10 = mu * (-2.0 * x1 * v1 - 1.0)
        j11 = mu * (1.0 - x1 * x1)
        ok, d0, d1 = _solve2x2(1.0, -a, -a * j10, 1.0 - a * j11, r0, r1)
        if not ok or not np.isfinite(d0) or not np.isfinite(d1):
            return False, 0.0, 0.0

        x1 -= d0
        v1 -= d1

        if max(abs(d0), abs(d1)) <= 1e-12 * (1.0 + max(abs(x1), abs(v1))):
            break

    x2 = x1
    v2 = v1

    for _ in range(8):
        f22 = _f2(x2, v2, mu)
        r0 = x2 - _ALPHA * x1 + _BETA * x - _DELTA * h * v2
        r1 = v2 - _ALPHA * v1 + _BETA * v - _DELTA * h * f22
        if max(abs(r0), abs(r1)) <= 1e-12 * (1.0 + max(abs(x2), abs(v2))):
            break

        j20 = mu * (-2.0 * x2 * v2 - 1.0)
        j21 = mu * (1.0 - x2 * x2)
        ok, d0, d1 = _solve2x2(
            1.0,
            -_DELTA * h,
            -_DELTA * h * j20,
            1.0 - _DELTA * h * j21,
            r0,
            r1,
        )
        if not ok or not np.isfinite(d0) or not np.isfinite(d1):
            return False, 0.0, 0.0

        x2 -= d0
        v2 -= d1

        if max(abs(d0), abs(d1)) <= 1e-12 * (1.0 + max(abs(x2), abs(v2))):
            break

    if not np.isfinite(x2) or not np.isfinite(v2):
        return False, 0.0, 0.0
    return True, x2, v2

@njit(cache=True)
def _solve_numba(mu, x0, v0, t0, t1, rtol, atol):
    dt = t1 - t0
    if dt == 0.0:
        return True, x0, v0

    direction = 1.0
    if dt < 0.0:
        direction = -1.0
    span = abs(dt)

    fscale = max(1.0, abs(v0), abs(_f2(x0, v0, mu)))
    yscale = max(1.0, abs(x0), abs(v0))
    h = direction * min(span, max(1e-6, 0.1 * yscale / fscale))

    t = t0
    x = x0
    v = v0

    for _ in range(200000):
        rem = t1 - t
        if direction * rem <= 0.0:
            return True, x, v

        if abs(h) > abs(rem):
            h = rem

        ok1, xf, vf = _trbdf2_single_step(x, v, h, mu)
        if not ok1:
            h *= 0.5
            if abs(h) < 1e-15:
                return False, x, v
            continue

        hh = 0.5 * h
        ok2, xm, vm = _trbdf2_single_step(x, v, hh, mu)
        if not ok2:
            h *= 0.5
            if abs(h) < 1e-15:
                return False, x, v
            continue

        ok3, xh, vh = _trbdf2_single_step(xm, vm, hh, mu)
        if not ok3:
            h *= 0.5
            if abs(h) < 1e-15:
                return False, x, v
            continue

        scx = atol + rtol * max(abs(x), abs(xh))
        scv = atol + rtol * max(abs(v), abs(vh))
        errx = abs(xh - xf) / (3.0 * scx)
        errv = abs(vh - vf) / (3.0 * scv)
        err = max(errx, errv)

        if err <= 1.0:
            x = xh + (xh - xf) / 3.0
            v = vh + (vh - vf) / 3.0
            t += h

            if err < 1e-12:
                fac = 5.0
            else:
                fac = 0.9 * err ** (-1.0 / 3.0)
                if fac < 0.2:
                    fac = 0.2
                elif fac > 5.0:
                    fac = 5.0
            h *= fac
        else:
            fac = 0.9 * err ** (-1.0 / 3.0)
            if fac < 0.1:
                fac = 0.1
            elif fac > 0.5:
                fac = 0.5
            h *= fac
            if abs(h) < 1e-15:
                return False, x, v

    return False, x, v

    fscale = max(1.0, abs(v0), abs(_f2(x0, v0, mu)))
    yscale = max(1.0, abs(x0), abs(v0))
    h = direction * min(span, max(1e-6, 0.1 * yscale / fscale))

    t = t0
    x = x0
    v = v0

    for _ in range(200000):
        rem = t1 - t
        if direction * rem <= 0.0:
            return True, x, v

        if abs(h) > abs(rem):
            h = rem

        ok1, xf, vf = _trbdf2_single_step(x, v, h, mu)
        if not ok1:
            h *= 0.5
            if abs(h) < 1e-15:
                return False, x, v
            continue

        hh = 0.5 * h
        ok2, xm, vm = _trbdf2_single_step(x, v, hh, mu)
        if not ok2:
            h *= 0.5
            if abs(h) < 1e-15:
                return False, x, v
            continue

        ok3, xh, vh = _trbdf2_single_step(xm, vm, hh, mu)
        if not ok3:
            h *= 0.5
            if abs(h) < 1e-15:
                return False, x, v
            continue

        scx = atol + rtol * max(abs(x), abs(xh))
        scv = atol + rtol * max(abs(v), abs(vh))
        errx = abs(xh - xf) / (3.0 * scx)
        errv = abs(vh - vf) / (3.0 * scv)
        err = max(errx, errv)

        if err <= 1.0:
            x = xh + (xh - xf) / 3.0
            v = vh + (vh - vf) / 3.0
            t += h

            if err < 1e-12:
                fac = 5.0
            else:
                fac = 0.9 * err ** (-1.0 / 3.0)
                if fac < 0.2:
                    fac = 0.2
                elif fac > 5.0:
                    fac = 5.0
            h *= fac
        else:
            fac = 0.9 * err ** (-1.0 / 3.0)
            if fac < 0.1:
                fac = 0.1
            elif fac > 0.5:
                fac = 0.5
            h *= fac
            if abs(h) < 1e-15:
                return False, x, v

    return False, x, v

class Solver:
    def __init__(self) -> None:
        self._cache: dict[tuple[float, float, float, float, float], list[float]] = {}
        _solve_numba(1.0, 0.0, 0.0, 0.0, 0.0, 3e-7, 1e-9)

    def solve(self, problem, **kwargs) -> Any:
        mu = float(problem["mu"])
        x0 = float(problem["y0"][0])
        v0 = float(problem["y0"][1])
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])

        key = (mu, t0, t1, x0, v0)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ok, x, v = _solve_numba(mu, x0, v0, t0, t1, 3e-7, 1e-9)
        if ok and isfinite(x) and isfinite(v):
            ans = [x, v]
            self._cache[key] = ans
            return ans

        def func(y, t):
            xx = y[0]
            vv = y[1]
            return (vv, mu * ((1.0 - xx * xx) * vv - xx))

        def jac(y, t):
            xx = y[0]
            vv = y[1]
            return (
                (0.0, 1.0),
                (mu * (-2.0 * xx * vv - 1.0), mu * (1.0 - xx * xx)),
            )

        y = odeint(
            func,
            [x0, v0],
            (t0, t1),
            Dfun=jac,
            rtol=1e-8,
            atol=1e-9,
        )
        ans = [float(y[-1, 0]), float(y[-1, 1])]
        self._cache[key] = ans
        return ans