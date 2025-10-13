from typing import Any, Dict, List
import numpy as np

# Try to use numba for speed; fall back gracefully if unavailable.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

# Dormand–Prince (RK45) coefficients (as used by SciPy's RK45) for 2D LV system
# c coefficients (nodes)
_C2 = 1.0 / 5.0
_C3 = 3.0 / 10.0
_C4 = 4.0 / 5.0
_C5 = 8.0 / 9.0
# a matrix (lower-triangular) rows
_A21 = 1.0 / 5.0

_A31 = 3.0 / 40.0
_A32 = 9.0 / 40.0

_A41 = 44.0 / 45.0
_A42 = -56.0 / 15.0
_A43 = 32.0 / 9.0

_A51 = 19372.0 / 6561.0
_A52 = -25360.0 / 2187.0
_A53 = 64448.0 / 6561.0
_A54 = -212.0 / 729.0

_A61 = 9017.0 / 3168.0
_A62 = -355.0 / 33.0
_A63 = 46732.0 / 5247.0
_A64 = 49.0 / 176.0
_A65 = -5103.0 / 18656.0

_A71 = 35.0 / 384.0
_A72 = 0.0
_A73 = 500.0 / 1113.0
_A74 = 125.0 / 192.0
_A75 = -2187.0 / 6784.0
_A76 = 11.0 / 84.0

# b (5th order solution) and b* (4th order) for error estimation
_B5_1 = 35.0 / 384.0
_B5_2 = 0.0
_B5_3 = 500.0 / 1113.0
_B5_4 = 125.0 / 192.0
_B5_5 = -2187.0 / 6784.0
_B5_6 = 11.0 / 84.0
_B5_7 = 0.0

_B4_1 = 5179.0 / 57600.0
_B4_2 = 0.0
_B4_3 = 7571.0 / 16695.0
_B4_4 = 393.0 / 640.0
_B4_5 = -92097.0 / 339200.0
_B4_6 = 187.0 / 2100.0
_B4_7 = 1.0 / 40.0

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _integrate_lv_numba(
        t0: float,
        t1: float,
        x0: float,
        y0: float,
        alpha: float,
        beta: float,
        delta: float,
        gamma: float,
        rtol: float,
        atol: float,
    ):
        # Returns (x_end, y_end, success_flag)
        t = t0
        x = x0
        y = y0
        direction = 1.0 if t1 >= t0 else -1.0
        total = t1 - t0

        # Initial derivative (k1)
        k1x = alpha * x - beta * x * y
        k1y = delta * x * y - gamma * y

        # Initial scales
        sx = atol + rtol * abs(x)
        sy = atol + rtol * abs(y)
        n = 2.0
        d0 = np.sqrt(((x / sx) * (x / sx) + (y / sy) * (y / sy)) / n)
        d1 = np.sqrt(((k1x / sx) * (k1x / sx) + (k1y / sy) * (k1y / sy)) / n)

        if d0 < 1e-5 or d1 < 1e-5:
            h = 1e-6
        else:
            h = 0.01 * d0 / d1

        # Bound initial step by total interval
        if abs(h) > abs(total):
            h = abs(total)
        h *= direction
        if h == 0.0:
            return x, y, True

        # FSAL: store last stage as first stage for next step
        have_prev_k = False
        prev_kx = 0.0
        prev_ky = 0.0

        max_steps = 10000000
        steps = 0

        while (t - t1) * direction + 1e-18 < 0.0:
            steps += 1
            if steps > max_steps:
                return x, y, False

            # Adjust step to not overshoot
            if (t + h - t1) * direction > 0.0:
                h = t1 - t
                if h == 0.0:
                    break

            # Compute stages
            if have_prev_k:
                k1x = prev_kx
                k1y = prev_ky
            else:
                k1x = alpha * x - beta * x * y
                k1y = delta * x * y - gamma * y

            # Stage 2
            xt = x + h * (_A21 * k1x)
            yt = y + h * (_A21 * k1y)
            k2x = alpha * xt - beta * xt * yt
            k2y = delta * xt * yt - gamma * yt

            # Stage 3
            xt = x + h * (_A31 * k1x + _A32 * k2x)
            yt = y + h * (_A31 * k1y + _A32 * k2y)
            k3x = alpha * xt - beta * xt * yt
            k3y = delta * xt * yt - gamma * yt

            # Stage 4
            xt = x + h * (_A41 * k1x + _A42 * k2x + _A43 * k3x)
            yt = y + h * (_A41 * k1y + _A42 * k2y + _A43 * k3y)
            k4x = alpha * xt - beta * xt * yt
            k4y = delta * xt * yt - gamma * yt

            # Stage 5
            xt = x + h * (_A51 * k1x + _A52 * k2x + _A53 * k3x + _A54 * k4x)
            yt = y + h * (_A51 * k1y + _A52 * k2y + _A53 * k3y + _A54 * k4y)
            k5x = alpha * xt - beta * xt * yt
            k5y = delta * xt * yt - gamma * yt

            # Stage 6
            xt = x + h * (_A61 * k1x + _A62 * k2x + _A63 * k3x + _A64 * k4x + _A65 * k5x)
            yt = y + h * (_A61 * k1y + _A62 * k2y + _A63 * k3y + _A64 * k4y + _A65 * k5y)
            k6x = alpha * xt - beta * xt * yt
            k6y = delta * xt * yt - gamma * yt

            # 5th order solution increment
            yx_new = x + h * (_B5_1 * k1x + _B5_2 * k2x + _B5_3 * k3x + _B5_4 * k4x + _B5_5 * k5x + _B5_6 * k6x)
            yy_new = y + h * (_B5_1 * k1y + _B5_2 * k2y + _B5_3 * k3y + _B5_4 * k4y + _B5_5 * k5y + _B5_6 * k6y)

            # Stage 7 for FSAL: f(t + h, y_new)
            k7x = alpha * yx_new - beta * yx_new * yy_new
            k7y = delta * yx_new * yy_new - gamma * yy_new

            # Error estimate using (b5 - b4)
            errx = h * (
                (_B5_1 - _B4_1) * k1x
                + (_B5_2 - _B4_2) * k2x
                + (_B5_3 - _B4_3) * k3x
                + (_B5_4 - _B4_4) * k4x
                + (_B5_5 - _B4_5) * k5x
                + (_B5_6 - _B4_6) * k6x
                + (_B5_7 - _B4_7) * k7x
            )
            erry = h * (
                (_B5_1 - _B4_1) * k1y
                + (_B5_2 - _B4_2) * k2y
                + (_B5_3 - _B4_3) * k3y
                + (_B5_4 - _B4_4) * k4y
                + (_B5_5 - _B4_5) * k5y
                + (_B5_6 - _B4_6) * k6y
                + (_B5_7 - _B4_7) * k7y
            )

            # Error norm
            sxn = atol + rtol * max(abs(x), abs(yx_new))
            syn = atol + rtol * max(abs(y), abs(yy_new))
            err_norm = np.sqrt(((errx / sxn) * (errx / sxn) + (erry / syn) * (erry / syn)) / n)

            if not np.isfinite(err_norm):
                # Reduce step aggressively if bad numbers
                h *= 0.2
                have_prev_k = False
                continue

            if err_norm <= 1.0:
                # Accept step
                t += h
                x = yx_new
                y = yy_new
                have_prev_k = True
                prev_kx = k7x  # FSAL
                prev_ky = k7y

                # Step size update
                if err_norm == 0.0:
                    factor = 5.0
                else:
                    factor = 0.9 * err_norm ** (-0.2)  # 1/order where order=5
                    if factor > 5.0:
                        factor = 5.0
                    elif factor < 0.2:
                        factor = 0.2
                h *= factor
            else:
                # Reject step, decrease h
                factor = max(0.2, 0.9 * err_norm ** (-0.25))  # slightly more conservative on reject
                h *= factor
                have_prev_k = False

            # Prevent tiny steps from stalling
            if t == t1:
                break
            if h == 0.0:
                h = 1e-16 * direction

        return x, y, True
else:
    _integrate_lv_numba = None  # type: ignore

# Fallback SciPy solver
def _solve_scipy(t0, t1, y0, params):
    from scipy.integrate import solve_ivp

    alpha = float(params["alpha"])
    beta = float(params["beta"])
    delta = float(params["delta"])
    gamma = float(params["gamma"])

    def f(_t, y):
        x, p = y
        dx_dt = alpha * x - beta * x * p
        dp_dt = delta * x * p - gamma * p
        return (dx_dt, dp_dt)

    sol = solve_ivp(
        f, (t0, t1), y0, method="DOP853", rtol=1e-10, atol=1e-10, dense_output=False
    )
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")
    return sol.y[:, -1]

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.asarray(problem["y0"], dtype=float)
        params = problem["params"]

        if t1 == t0:
            return y0.tolist()

        alpha = float(params["alpha"])
        beta = float(params["beta"])
        delta = float(params["delta"])
        gamma = float(params["gamma"])

        # Tight tolerances to match the reference within required tolerance
        rtol = 1e-10
        atol = 1e-10

        # Prefer numba-accelerated integrator if available
        if NUMBA_AVAILABLE and _integrate_lv_numba is not None:
            x_end, y_end, ok = _integrate_lv_numba(
                t0, t1, float(y0[0]), float(y0[1]), alpha, beta, delta, gamma, rtol, atol
            )
            if ok and np.isfinite(x_end) and np.isfinite(y_end):
                # Do not clip unless tiny negative due to numerical noise
                return [x_end, y_end]
            # Fall back to SciPy if numba path failed
        y_final = _solve_scipy(t0, t1, y0, params)
        return y_final.tolist()