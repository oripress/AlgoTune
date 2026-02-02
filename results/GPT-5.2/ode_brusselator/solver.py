from __future__ import annotations

from typing import Any

import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None

if nb is not None:

    @nb.njit(cache=True, fastmath=True)
    def _rhs_brusselator(x: float, y: float, A: float, B: float) -> tuple[float, float]:
        # dX/dt = A + X^2 Y - (B+1) X
        # dY/dt = B X - X^2 Y
        x2y = x * x * y
        dx = A + x2y - (B + 1.0) * x
        dy = B * x - x2y
        return dx, dy

    @nb.njit(cache=True, fastmath=True)
    def _integrate_brusselator_dopri5(
        t0: float,
        t1: float,
        x0: float,
        y0: float,
        A: float,
        B: float,
        rtol: float,
        atol: float,
    ) -> tuple[float, float]:
        # Dormand–Prince 5(4) method (same family as RK45)
        if t1 == t0:
            return x0, y0

        t = t0
        x = x0
        y = y0

        # Initial derivative (also used for initial step size heuristic and FSAL)
        fx0, fy0 = _rhs_brusselator(x, y, A, B)

        # Heuristic initial step size (cheap and robust)
        scx = atol + rtol * abs(x)
        scy = atol + rtol * abs(y)
        d0 = max(abs(x) / scx, abs(y) / scy)
        d1 = max(abs(fx0) / scx, abs(fy0) / scy)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        # Second derivative estimate via one Euler step
        xe = x + h0 * fx0
        ye = y + h0 * fy0
        fx1, fy1 = _rhs_brusselator(xe, ye, A, B)
        d2 = max(abs(fx1 - fx0) / scx, abs(fy1 - fy0) / scy) / max(h0, 1e-16)

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** 0.2

        h = min(100.0 * h0, h1)
        # Cap step size to avoid huge initial jumps; dynamics timescale ~ O(1)
        if h > 0.25:
            h = 0.25

        direction = 1.0
        if t1 < t0:
            direction = -1.0
            h = -abs(h)

        # FSAL: k1 at current state is reused between accepted steps
        k1x = fx0
        k1y = fy0

        # Coefficients (hard-coded for speed)
        a21 = 0.2

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

        # b (5th order)
        b1 = 35.0 / 384.0
        b3 = 500.0 / 1113.0
        b4 = 125.0 / 192.0
        b5 = -2187.0 / 6784.0
        b6 = 11.0 / 84.0

        # bhat (4th order)
        bh1 = 5179.0 / 57600.0
        bh3 = 7571.0 / 16695.0
        bh4 = 393.0 / 640.0
        bh5 = -92097.0 / 339200.0
        bh6 = 187.0 / 2100.0
        bh7 = 1.0 / 40.0

        # Safety parameters
        safety = 0.9
        min_fac = 0.2
        max_fac = 5.0
        # Exponent for order 5
        exp = -0.2

        # Limit steps to prevent pathological infinite loops
        max_steps = 2_000_000
        n_steps = 0

        while (t - t1) * direction < 0.0:
            n_steps += 1
            if n_steps > max_steps:
                break

            # clamp to final time
            if (t + h - t1) * direction > 0.0:
                h = t1 - t

            # Stage 1
            k1x, k1y = _rhs_brusselator(x, y, A, B)

            # Stage 2
            x2 = x + h * a21 * k1x
            y2 = y + h * a21 * k1y
            k2x, k2y = _rhs_brusselator(x2, y2, A, B)

            # Stage 3
            x3 = x + h * (a31 * k1x + a32 * k2x)
            y3 = y + h * (a31 * k1y + a32 * k2y)
            k3x, k3y = _rhs_brusselator(x3, y3, A, B)

            # Stage 4
            x4 = x + h * (a41 * k1x + a42 * k2x + a43 * k3x)
            y4 = y + h * (a41 * k1y + a42 * k2y + a43 * k3y)
            k4x, k4y = _rhs_brusselator(x4, y4, A, B)

            # Stage 5
            x5 = x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x)
            y5 = y + h * (a51 * k1y + a52 * k2y + a53 * k3y + a54 * k4y)
            k5x, k5y = _rhs_brusselator(x5, y5, A, B)

            # Stage 6
            x6 = x + h * (
                a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x
            )
            y6 = y + h * (
                a61 * k1y + a62 * k2y + a63 * k3y + a64 * k4y + a65 * k5y
            )
            k6x, k6y = _rhs_brusselator(x6, y6, A, B)

            # Stage 7 uses b's (k7 at y5th)
            x7 = x + h * (b1 * k1x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x)
            y7 = y + h * (b1 * k1y + b3 * k3y + b4 * k4y + b5 * k5y + b6 * k6y)
            k7x, k7y = _rhs_brusselator(x7, y7, A, B)

            # 5th order solution (y7 is already it)
            x_new = x7
            y_new = y7

            # 4th order solution for error estimate
            x4th = x + h * (
                bh1 * k1x
                + bh3 * k3x
                + bh4 * k4x
                + bh5 * k5x
                + bh6 * k6x
                + bh7 * k7x
            )
            y4th = y + h * (
                bh1 * k1y
                + bh3 * k3y
                + bh4 * k4y
                + bh5 * k5y
                + bh6 * k6y
                + bh7 * k7y
            )

            ex = x_new - x4th
            ey = y_new - y4th

            scx = atol + rtol * max(abs(x), abs(x_new))
            scy = atol + rtol * max(abs(y), abs(y_new))
            err = max(abs(ex) / scx, abs(ey) / scy)

            if err <= 1.0 or abs(h) <= 1e-15:
                # accept
                t += h
                x = x_new
                y = y_new

                if err == 0.0:
                    fac = max_fac
                else:
                    fac = safety * (err**exp)
                    if fac < min_fac:
                        fac = min_fac
                    elif fac > max_fac:
                        fac = max_fac
                h *= fac
                # Keep h sane
                if abs(h) > 1.0:
                    h = direction * 1.0
            else:
                # reject, reduce step
                fac = safety * (err**exp)
                if fac < 0.1:
                    fac = 0.1
                h *= fac

        return x, y

class Solver:
    def __init__(self) -> None:
        # Trigger numba compilation outside solve() so it won't count in runtime.
        if nb is not None:
            _integrate_brusselator_dopri5(0.0, 0.1, 1.1, 3.2, 1.0, 3.0, 1e-8, 1e-8)

    def solve(self, problem: dict, **kwargs) -> Any:
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        A = float(params["A"])
        B = float(params["B"])

        x0 = float(y0[0])
        y00 = float(y0[1])

        # Match reference tolerances more closely to reduce mismatch rate.
        rtol = 1e-8
        atol = 1e-8

        if nb is None:
            # Fallback: RK4 fixed-step (conservative step count for accuracy).
            if t1 == t0:
                return [x0, y00]

            n_steps = 50000
            dt = (t1 - t0) / float(n_steps)
            x = x0
            y = y00
            for _ in range(n_steps):
                k1x, k1y = _rhs_py(x, y, A, B)
                k2x, k2y = _rhs_py(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, A, B)
                k3x, k3y = _rhs_py(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, A, B)
                k4x, k4y = _rhs_py(x + dt * k3x, y + dt * k3y, A, B)
                x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
                y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            return [x, y]

        xf, yf = _integrate_brusselator_dopri5(t0, t1, x0, y00, A, B, rtol, atol)
        return [xf, yf]

def _rhs_py(x: float, y: float, A: float, B: float) -> tuple[float, float]:
    x2y = x * x * y
    dx = A + x2y - (B + 1.0) * x
    dy = B * x - x2y
    return dx, dy