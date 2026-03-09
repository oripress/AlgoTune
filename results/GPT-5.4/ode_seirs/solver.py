from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _rk45_solve_reduced(
    t0: float,
    t1: float,
    s: float,
    e: float,
    i: float,
    beta: float,
    sigma: float,
    gamma: float,
    omega: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, float]:
    if t1 == t0:
        total = s + e + i
        r = 1.0 - total
        total4 = total + r
        if total4 != 0.0:
            inv = 1.0 / total4
            return s * inv, e * inv, i * inv
        return s, e, i

    t = t0
    direction = 1.0 if t1 >= t0 else -1.0
    dt_total = t1 - t0
    adt = abs(dt_total)

    max_rate = beta + sigma + gamma + omega
    if max_rate < 1e-12:
        max_rate = 1e-12
    h = min(adt, max(1e-3, 0.06 / max_rate)) * direction

    min_h = 1e-14 * max(1.0, adt)
    max_h = adt

    r = 1.0 - s - e - i
    inf = beta * s * i
    k1s = -inf + omega * r
    k1e = inf - sigma * e
    k1i = sigma * e - gamma * i
    have_k1 = True

    while direction * (t1 - t) > 0.0:
        ah = abs(h)
        if ah < min_h:
            h = direction * min_h
        elif ah > max_h:
            h = direction * max_h
        if direction * (t + h - t1) > 0.0:
            h = t1 - t

        if not have_k1:
            r = 1.0 - s - e - i
            inf = beta * s * i
            k1s = -inf + omega * r
            k1e = inf - sigma * e
            k1i = sigma * e - gamma * i

        s2 = s + h * (1.0 / 5.0) * k1s
        e2 = e + h * (1.0 / 5.0) * k1e
        i2 = i + h * (1.0 / 5.0) * k1i
        r2 = 1.0 - s2 - e2 - i2
        inf2 = beta * s2 * i2
        k2s = -inf2 + omega * r2
        k2e = inf2 - sigma * e2
        k2i = sigma * e2 - gamma * i2

        s3 = s + h * ((3.0 / 40.0) * k1s + (9.0 / 40.0) * k2s)
        e3 = e + h * ((3.0 / 40.0) * k1e + (9.0 / 40.0) * k2e)
        i3 = i + h * ((3.0 / 40.0) * k1i + (9.0 / 40.0) * k2i)
        r3 = 1.0 - s3 - e3 - i3
        inf3 = beta * s3 * i3
        k3s = -inf3 + omega * r3
        k3e = inf3 - sigma * e3
        k3i = sigma * e3 - gamma * i3

        s4 = s + h * ((44.0 / 45.0) * k1s + (-56.0 / 15.0) * k2s + (32.0 / 9.0) * k3s)
        e4 = e + h * ((44.0 / 45.0) * k1e + (-56.0 / 15.0) * k2e + (32.0 / 9.0) * k3e)
        i4 = i + h * ((44.0 / 45.0) * k1i + (-56.0 / 15.0) * k2i + (32.0 / 9.0) * k3i)
        r4 = 1.0 - s4 - e4 - i4
        inf4 = beta * s4 * i4
        k4s = -inf4 + omega * r4
        k4e = inf4 - sigma * e4
        k4i = sigma * e4 - gamma * i4

        s5 = s + h * (
            (19372.0 / 6561.0) * k1s
            + (-25360.0 / 2187.0) * k2s
            + (64448.0 / 6561.0) * k3s
            + (-212.0 / 729.0) * k4s
        )
        e5 = e + h * (
            (19372.0 / 6561.0) * k1e
            + (-25360.0 / 2187.0) * k2e
            + (64448.0 / 6561.0) * k3e
            + (-212.0 / 729.0) * k4e
        )
        i5 = i + h * (
            (19372.0 / 6561.0) * k1i
            + (-25360.0 / 2187.0) * k2i
            + (64448.0 / 6561.0) * k3i
            + (-212.0 / 729.0) * k4i
        )
        r5 = 1.0 - s5 - e5 - i5
        inf5 = beta * s5 * i5
        k5s = -inf5 + omega * r5
        k5e = inf5 - sigma * e5
        k5i = sigma * e5 - gamma * i5

        s6 = s + h * (
            (9017.0 / 3168.0) * k1s
            + (-355.0 / 33.0) * k2s
            + (46732.0 / 5247.0) * k3s
            + (49.0 / 176.0) * k4s
            + (-5103.0 / 18656.0) * k5s
        )
        e6 = e + h * (
            (9017.0 / 3168.0) * k1e
            + (-355.0 / 33.0) * k2e
            + (46732.0 / 5247.0) * k3e
            + (49.0 / 176.0) * k4e
            + (-5103.0 / 18656.0) * k5e
        )
        i6 = i + h * (
            (9017.0 / 3168.0) * k1i
            + (-355.0 / 33.0) * k2i
            + (46732.0 / 5247.0) * k3i
            + (49.0 / 176.0) * k4i
            + (-5103.0 / 18656.0) * k5i
        )
        r6 = 1.0 - s6 - e6 - i6
        inf6 = beta * s6 * i6
        k6s = -inf6 + omega * r6
        k6e = inf6 - sigma * e6
        k6i = sigma * e6 - gamma * i6

        y5s = s + h * (
            (35.0 / 384.0) * k1s
            + (500.0 / 1113.0) * k3s
            + (125.0 / 192.0) * k4s
            + (-2187.0 / 6784.0) * k5s
            + (11.0 / 84.0) * k6s
        )
        y5e = e + h * (
            (35.0 / 384.0) * k1e
            + (500.0 / 1113.0) * k3e
            + (125.0 / 192.0) * k4e
            + (-2187.0 / 6784.0) * k5e
            + (11.0 / 84.0) * k6e
        )
        y5i = i + h * (
            (35.0 / 384.0) * k1i
            + (500.0 / 1113.0) * k3i
            + (125.0 / 192.0) * k4i
            + (-2187.0 / 6784.0) * k5i
            + (11.0 / 84.0) * k6i
        )

        r7 = 1.0 - y5s - y5e - y5i
        inf7 = beta * y5s * y5i
        k7s = -inf7 + omega * r7
        k7e = inf7 - sigma * y5e
        k7i = sigma * y5e - gamma * y5i

        y4s = s + h * (
            (5179.0 / 57600.0) * k1s
            + (7571.0 / 16695.0) * k3s
            + (393.0 / 640.0) * k4s
            + (-92097.0 / 339200.0) * k5s
            + (187.0 / 2100.0) * k6s
            + (1.0 / 40.0) * k7s
        )
        y4e = e + h * (
            (5179.0 / 57600.0) * k1e
            + (7571.0 / 16695.0) * k3e
            + (393.0 / 640.0) * k4e
            + (-92097.0 / 339200.0) * k5e
            + (187.0 / 2100.0) * k6e
            + (1.0 / 40.0) * k7e
        )
        y4i = i + h * (
            (5179.0 / 57600.0) * k1i
            + (7571.0 / 16695.0) * k3i
            + (393.0 / 640.0) * k4i
            + (-92097.0 / 339200.0) * k5i
            + (187.0 / 2100.0) * k6i
            + (1.0 / 40.0) * k7i
        )

        errs = abs(y5s - y4s) / (atol + rtol * max(abs(s), abs(y5s)))
        erre = abs(y5e - y4e) / (atol + rtol * max(abs(e), abs(y5e)))
        erri = abs(y5i - y4i) / (atol + rtol * max(abs(i), abs(y5i)))
        err = max(errs, erre, erri)

        if err <= 1.0:
            s = y5s
            e = y5e
            i = y5i
            t += h
            k1s = k7s
            k1e = k7e
            k1i = k7i
            have_k1 = True

            if err == 0.0:
                fac = 4.0
            else:
                fac = 0.95 * err ** (-0.2)
                if fac > 4.0:
                    fac = 4.0
            h *= fac
        else:
            fac = 0.95 * err ** (-0.25)
            if fac < 0.15:
                fac = 0.15
            h *= fac
            have_k1 = False

    return s, e, i

class Solver:
    def __init__(self) -> None:
        _rk45_solve_reduced(0.0, 1.0, 0.89, 0.01, 0.005, 0.35, 0.2, 0.1, 0.002, 2e-6, 2e-8)

    def solve(self, problem, **kwargs) -> Any:
        y0 = problem["y0"]
        params = problem["params"]

        s, e, i = _rk45_solve_reduced(
            float(problem["t0"]),
            float(problem["t1"]),
            float(y0[0]),
            float(y0[1]),
            float(y0[2]),
            float(params["beta"]),
            float(params["sigma"]),
            float(params["gamma"]),
            float(params["omega"]),
            2e-6,
            2e-8,
        )
        r = 1.0 - s - e - i
        return [s, e, i, r]