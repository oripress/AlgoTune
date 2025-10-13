from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        # Parse input
        y0_list = problem["y0"]
        if isinstance(y0_list, (list, tuple, np.ndarray)):
            if len(y0_list) != 2:
                raise ValueError("y0 must be of length 2 for the Brusselator model.")
            x0 = float(y0_list[0])
            y0 = float(y0_list[1])
        else:
            raise ValueError("y0 must be a list/tuple/array of two floats.")

        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        A = float(params["A"])
        B = float(params["B"])

        # Tolerances (match reference)
        rtol = float(kwargs.get("rtol", 1e-8))
        atol = float(kwargs.get("atol", 1e-8))

        # Early exit if no integration needed
        if t1 == t0:
            return np.array([x0, y0], dtype=float)

        # Direction
        direction = 1.0 if t1 > t0 else -1.0

        # Right-hand side of the Brusselator
        def f_eval(x: float, y: float) -> Tuple[float, float]:
            # dX/dt = A + X^2 Y - (B+1) X
            # dY/dt = B X - X^2 Y
            xx = x * x
            xy = xx * y
            dx = A + xy - (B + 1.0) * x
            dy = B * x - xy
            return dx, dy

        # Compute initial derivative
        k1x, k1y = f_eval(x0, y0)

        # Helper: error scale
        def scale_vals(x: float, y: float, xn: float, yn: float) -> Tuple[float, float]:
            sx = atol + rtol * max(abs(x), abs(xn))
            sy = atol + rtol * max(abs(y), abs(yn))
            return sx, sy

        # Initial step size estimation (Hairer-like)
        def initial_step(
            t: float,
            x: float,
            y: float,
            fx: float,
            fy: float,
            t_end: float,
        ) -> float:
            # Weighted RMS norms
            sx0 = atol + rtol * abs(x)
            sy0 = atol + rtol * abs(y)
            d0 = np.sqrt((x / sx0) ** 2 + (y / sy0) ** 2) / np.sqrt(2.0)
            d1 = np.sqrt((fx / sx0) ** 2 + (fy / sy0) ** 2) / np.sqrt(2.0)

            if d0 < 1e-5 or d1 < 1e-5:
                h0 = 1e-6
            else:
                h0 = 0.01 * d0 / d1

            h0 = min(abs(t_end - t), h0)
            h_try = direction * h0

            # One-step Euler to estimate second derivative
            x1 = x + h_try * fx
            y1 = y + h_try * fy
            f1x, f1y = f_eval(x1, y1)
            sx1 = atol + rtol * abs(x1)
            sy1 = atol + rtol * abs(y1)
            d2 = np.sqrt(((f1x - fx) / sx1) ** 2 + ((f1y - fy) / sy1) ** 2) / (np.sqrt(2.0) * abs(h0))
            dmax = max(d1, d2)
            if dmax <= 1e-15:
                h1 = max(1e-6, h0 * 1e-3)
            else:
                h1 = 0.01 * (1.0 / dmax) ** (1.0 / 5.0)

            h = min(100.0 * h0, h1)
            h = min(abs(t_end - t), h)
            return direction * h

        # Dormand–Prince RK45 coefficients (DOPRI5) with FSAL
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

        # 7th stage to leverage FSAL (y7 equals y5)
        a71 = 35.0 / 384.0
        a72 = 0.0
        a73 = 500.0 / 1113.0
        a74 = 125.0 / 192.0
        a75 = -2187.0 / 6784.0
        a76 = 11.0 / 84.0

        # 5th order solution weights (b) - note b7 = 0
        b1 = 35.0 / 384.0
        b2 = 0.0
        b3 = 500.0 / 1113.0
        b4 = 125.0 / 192.0
        b5 = -2187.0 / 6784.0
        b6 = 11.0 / 84.0
        # b7 = 0

        # 4th order solution weights (bhat) for error estimate
        bh1 = 5179.0 / 57600.0
        bh2 = 0.0
        bh3 = 7571.0 / 16695.0
        bh4 = 393.0 / 640.0
        bh5 = -92097.0 / 339200.0
        bh6 = 187.0 / 2100.0
        bh7 = 1.0 / 40.0

        # Parameters for adaptive step size control
        safety = 0.9
        min_factor = 0.2
        max_factor = 10.0
        order = 5.0

        # Initialize
        t = t0
        x = x0
        y = y0
        fx = k1x
        fy = k1y

        h = initial_step(t, x, y, fx, fy, t1)
        # Ensure non-zero step
        if h == 0.0:
            h = direction * 1e-8

        # Main integration loop
        max_steps = 10_000_000
        steps = 0
        t_end = t1

        while steps < max_steps:
            steps += 1

            # Adjust step to not overshoot
            h_remaining = t_end - t
            if direction * h_remaining <= 0.0:
                break
            if abs(h) > abs(h_remaining):
                h = h_remaining

            # Stage 1: k1 already known (FSAL variable: fx, fy)
            k1x, k1y = fx, fy

            # Stage 2
            x2 = x + h * (a21 * k1x)
            y2 = y + h * (a21 * k1y)
            k2x, k2y = f_eval(x2, y2)

            # Stage 3
            x3 = x + h * (a31 * k1x + a32 * k2x)
            y3 = y + h * (a31 * k1y + a32 * k2y)
            k3x, k3y = f_eval(x3, y3)

            # Stage 4
            x4 = x + h * (a41 * k1x + a42 * k2x + a43 * k3x)
            y4 = y + h * (a41 * k1y + a42 * k2y + a43 * k3y)
            k4x, k4y = f_eval(x4, y4)

            # Stage 5
            x5 = x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x)
            y5 = y + h * (a51 * k1y + a52 * k2y + a53 * k3y + a54 * k4y)
            k5x, k5y = f_eval(x5, y5)

            # Stage 6
            x6 = x + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x)
            y6 = y + h * (a61 * k1y + a62 * k2y + a63 * k3y + a64 * k4y + a65 * k5y)
            k6x, k6y = f_eval(x6, y6)

            # 5th order solution (y_next)
            xn = x + h * (b1 * k1x + b2 * k2x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x)
            yn = y + h * (b1 * k1y + b2 * k2y + b3 * k3y + b4 * k4y + b5 * k5y + b6 * k6y)

            # Stage 7 (equals evaluation at y_next due to FSAL)
            x7 = x + h * (a71 * k1x + a72 * k2x + a73 * k3x + a74 * k4x + a75 * k5x + a76 * k6x)
            y7 = y + h * (a71 * k1y + a72 * k2y + a73 * k3y + a74 * k4y + a75 * k5y + a76 * k6y)
            # Due to FSAL, (x7, y7) == (xn, yn) theoretically; numeric differences could be tiny.

            k7x, k7y = f_eval(x7, y7)

            # 4th order solution for error estimate
            xr = x + h * (bh1 * k1x + bh2 * k2x + bh3 * k3x + bh4 * k4x + bh5 * k5x + bh6 * k6x + bh7 * k7x)
            yr = y + h * (bh1 * k1y + bh2 * k2y + bh3 * k3y + bh4 * k4y + bh5 * k5y + bh6 * k6y + bh7 * k7y)

            # Error estimate e = y5 - y4 (xn,yn) - (xr,yr)
            ex = xn - xr
            ey = yn - yr

            # Scaled error norm (infinity norm)
            sx, sy = scale_vals(x, y, xn, yn)
            err = max(abs(ex) / sx, abs(ey) / sy)

            if err <= 1.0:
                # Accept step
                t += h
                x = xn
                y = yn

                # FSAL: reuse k7 for next step's k1
                fx, fy = k7x, k7y

                # Adapt step
                if err == 0.0:
                    factor = max_factor
                else:
                    factor = safety * err ** (-1.0 / order)
                    factor = min(max_factor, max(min_factor, factor))
                h *= factor
            else:
                # Reject step, reduce h and retry; do not advance t
                factor = safety * err ** (-1.0 / order)
                factor = min(1.0, max(min_factor, factor))
                h *= factor

                # Recompute k1 at current state only if not using FSAL (we already have fx, fy)
                # Here fx, fy correspond to f(x, y) at current t, so nothing to do.

                continue

        # Return final state
        return np.array([x, y], dtype=float)