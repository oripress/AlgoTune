from __future__ import annotations

from typing import Any, Callable, Optional

import math

class Solver:
    """
    Lotka–Volterra final-state solver.

    Goal: match a high-accuracy SciPy solve_ivp reference (RK45, rtol=atol=1e-10)
    but faster. We integrate in log-space (u=log x, v=log y) to ensure positivity:

        du/dt = alpha - beta*exp(v)
        dv/dt = delta*exp(u) - gamma

    Uses an adaptive Dormand–Prince RK5(4) method (DOPRI5). If numba is available,
    a JIT-compiled version is used for significantly lower overhead.
    """

    _nb_fun: Optional[Callable[..., Any]] = None

    def __init__(self) -> None:
        # Compile numba kernel during init (init time is not counted).
        if Solver._nb_fun is not None:
            return
        try:
            from numba import njit  # type: ignore

            @njit(cache=True, fastmath=True)
            def _dopri5_nb(
                t0: float,
                t1: float,
                x0: float,
                y0: float,
                alpha: float,
                beta: float,
                delta: float,
                gamma: float,
            ) -> tuple[float, float]:
                if t1 == t0:
                    return x0, y0

                if x0 <= 0.0 or y0 <= 0.0:
                    # Cannot log-transform; model keeps zeros at zero.
                    if x0 < 0.0:
                        x0 = 0.0
                    if y0 < 0.0:
                        y0 = 0.0
                    return x0, y0

                exp = math.exp
                log = math.log
                abs_ = abs

                u = log(x0)
                v = log(y0)

                dt_total = abs_(t1 - t0)
                # Tight tolerances to match high-accuracy reference (phase matters).
                rtol = 1e-10
                atol = 1e-12

                safety = 0.9
                fac_min = 0.2
                fac_max = 5.0
                pow_ = 0.2  # 1/5

                max_rate = max(abs_(alpha) + abs_(beta * y0), abs_(gamma) + abs_(delta * x0), 1e-12)
                h = min(dt_total, 0.25 / max_rate, 0.25)
                if h < 1e-8:
                    h = 1e-8

                sign = 1.0 if t1 >= t0 else -1.0
                t = t0

                # DOPRI5 coefficients
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

                b1 = 35.0 / 384.0
                b3 = 500.0 / 1113.0
                b4 = 125.0 / 192.0
                b5 = -2187.0 / 6784.0
                b6 = 11.0 / 84.0

                # (b - b_hat)
                e1 = b1 - 5179.0 / 57600.0
                e3 = b3 - 7571.0 / 16695.0
                e4 = b4 - 393.0 / 640.0
                e5 = b5 - (-92097.0 / 339200.0)
                e6 = b6 - 187.0 / 2100.0
                e7 = -1.0 / 40.0

                h_min = 1e-10
                max_iter = 2_000_000
                it = 0

                while sign * (t1 - t) > 0.0:
                    it += 1
                    if it > max_iter:
                        break

                    remaining = sign * (t1 - t)
                    h_abs = h if h < remaining else remaining
                    h_signed = sign * h_abs

                    # k1
                    k1u = alpha - beta * exp(v)
                    k1v = delta * exp(u) - gamma

                    # k2
                    u2 = u + h_signed * (a21 * k1u)
                    v2 = v + h_signed * (a21 * k1v)
                    k2u = alpha - beta * exp(v2)
                    k2v = delta * exp(u2) - gamma

                    # k3
                    u3 = u + h_signed * (a31 * k1u + a32 * k2u)
                    v3 = v + h_signed * (a31 * k1v + a32 * k2v)
                    k3u = alpha - beta * exp(v3)
                    k3v = delta * exp(u3) - gamma

                    # k4
                    u4 = u + h_signed * (a41 * k1u + a42 * k2u + a43 * k3u)
                    v4 = v + h_signed * (a41 * k1v + a42 * k2v + a43 * k3v)
                    k4u = alpha - beta * exp(v4)
                    k4v = delta * exp(u4) - gamma

                    # k5
                    u5 = u + h_signed * (a51 * k1u + a52 * k2u + a53 * k3u + a54 * k4u)
                    v5 = v + h_signed * (a51 * k1v + a52 * k2v + a53 * k3v + a54 * k4v)
                    k5u = alpha - beta * exp(v5)
                    k5v = delta * exp(u5) - gamma

                    # k6
                    u6 = u + h_signed * (a61 * k1u + a62 * k2u + a63 * k3u + a64 * k4u + a65 * k5u)
                    v6 = v + h_signed * (a61 * k1v + a62 * k2v + a63 * k3v + a64 * k4v + a65 * k5v)
                    k6u = alpha - beta * exp(v6)
                    k6v = delta * exp(u6) - gamma

                    # 5th order solution
                    u7 = u + h_signed * (b1 * k1u + b3 * k3u + b4 * k4u + b5 * k5u + b6 * k6u)
                    v7 = v + h_signed * (b1 * k1v + b3 * k3v + b4 * k4v + b5 * k5v + b6 * k6v)
                    k7u = alpha - beta * exp(v7)
                    k7v = delta * exp(u7) - gamma

                    # error
                    err_u = h_signed * (e1 * k1u + e3 * k3u + e4 * k4u + e5 * k5u + e6 * k6u + e7 * k7u)
                    err_v = h_signed * (e1 * k1v + e3 * k3v + e4 * k4v + e5 * k5v + e6 * k6v + e7 * k7v)

                    su = atol + rtol * max(1.0, abs_(u), abs_(u7))
                    sv = atol + rtol * max(1.0, abs_(v), abs_(v7))
                    err = max(abs_(err_u) / su, abs_(err_v) / sv)

                    accept = (err <= 1.0) or (h_abs <= h_min * 1.01)
                    if accept:
                        u = u7
                        v = v7
                        t += h_signed

                    # update h
                    if err == 0.0:
                        fac = fac_max
                    else:
                        fac = safety * (err ** (-pow_))
                        if fac < fac_min:
                            fac = fac_min
                        elif fac > fac_max:
                            fac = fac_max

                    h = h_abs * fac
                    if h < h_min:
                        h = h_min
                    elif h > 1.0:
                        h = 1.0

                xf = exp(u)
                yf = exp(v)
                if xf < 0.0:
                    xf = 0.0
                if yf < 0.0:
                    yf = 0.0
                return xf, yf

            # Force compilation in init.
            _ = _dopri5_nb(0.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 1.0)
            Solver._nb_fun = _dopri5_nb
        except Exception:
            Solver._nb_fun = None

    @staticmethod
    def _dopri5_py(
        t0: float,
        t1: float,
        x0: float,
        y0: float,
        alpha: float,
        beta: float,
        delta: float,
        gamma: float,
    ) -> list[float]:
        if t1 == t0:
            return [float(x0), float(y0)]

        if x0 <= 0.0 or y0 <= 0.0:
            if x0 < 0.0:
                x0 = 0.0
            if y0 < 0.0:
                y0 = 0.0
            return [float(x0), float(y0)]

        exp = math.exp
        log = math.log
        abs_ = abs
        max_ = max
        min_ = min

        u = log(x0)
        v = log(y0)

        dt_total = abs_(t1 - t0)
        rtol = 1e-10
        atol = 1e-12

        safety = 0.9
        fac_min = 0.2
        fac_max = 5.0
        pow_ = 0.2

        max_rate = max_(abs_(alpha) + abs_(beta * y0), abs_(gamma) + abs_(delta * x0), 1e-12)
        h = min_(dt_total, 0.25 / max_rate, 0.25)
        if h < 1e-8:
            h = 1e-8

        sign = 1.0 if t1 >= t0 else -1.0
        t = t0

        # DOPRI5 coeffs
        a21 = 1.0 / 5.0
        a31, a32 = 3.0 / 40.0, 9.0 / 40.0
        a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
        a51, a52, a53, a54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
        a61, a62, a63, a64, a65 = (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        )

        b1, b3, b4, b5, b6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0

        e1 = b1 - 5179.0 / 57600.0
        e3 = b3 - 7571.0 / 16695.0
        e4 = b4 - 393.0 / 640.0
        e5 = b5 - (-92097.0 / 339200.0)
        e6 = b6 - 187.0 / 2100.0
        e7 = -1.0 / 40.0

        h_min = 1e-10
        max_iter = 2_000_000
        it = 0

        while sign * (t1 - t) > 0.0:
            it += 1
            if it > max_iter:
                break

            remaining = sign * (t1 - t)
            h_abs = h if h < remaining else remaining
            h_signed = sign * h_abs

            k1u = alpha - beta * exp(v)
            k1v = delta * exp(u) - gamma

            u2 = u + h_signed * (a21 * k1u)
            v2 = v + h_signed * (a21 * k1v)
            k2u = alpha - beta * exp(v2)
            k2v = delta * exp(u2) - gamma

            u3 = u + h_signed * (a31 * k1u + a32 * k2u)
            v3 = v + h_signed * (a31 * k1v + a32 * k2v)
            k3u = alpha - beta * exp(v3)
            k3v = delta * exp(u3) - gamma

            u4 = u + h_signed * (a41 * k1u + a42 * k2u + a43 * k3u)
            v4 = v + h_signed * (a41 * k1v + a42 * k2v + a43 * k3v)
            k4u = alpha - beta * exp(v4)
            k4v = delta * exp(u4) - gamma

            u5 = u + h_signed * (a51 * k1u + a52 * k2u + a53 * k3u + a54 * k4u)
            v5 = v + h_signed * (a51 * k1v + a52 * k2v + a53 * k3v + a54 * k4v)
            k5u = alpha - beta * exp(v5)
            k5v = delta * exp(u5) - gamma

            u6 = u + h_signed * (a61 * k1u + a62 * k2u + a63 * k3u + a64 * k4u + a65 * k5u)
            v6 = v + h_signed * (a61 * k1v + a62 * k2v + a63 * k3v + a64 * k4v + a65 * k5v)
            k6u = alpha - beta * exp(v6)
            k6v = delta * exp(u6) - gamma

            u7 = u + h_signed * (b1 * k1u + b3 * k3u + b4 * k4u + b5 * k5u + b6 * k6u)
            v7 = v + h_signed * (b1 * k1v + b3 * k3v + b4 * k4v + b5 * k5v + b6 * k6v)
            k7u = alpha - beta * exp(v7)
            k7v = delta * exp(u7) - gamma

            err_u = h_signed * (e1 * k1u + e3 * k3u + e4 * k4u + e5 * k5u + e6 * k6u + e7 * k7u)
            err_v = h_signed * (e1 * k1v + e3 * k3v + e4 * k4v + e5 * k5v + e6 * k6v + e7 * k7v)

            su = atol + rtol * max_(1.0, abs_(u), abs_(u7))
            sv = atol + rtol * max_(1.0, abs_(v), abs_(v7))
            err = max_(abs_(err_u) / su, abs_(err_v) / sv)

            accept = (err <= 1.0) or (h_abs <= h_min * 1.01)
            if accept:
                u = u7
                v = v7
                t += h_signed

            if err == 0.0:
                fac = fac_max
            else:
                fac = safety * (err ** (-pow_))
                if fac < fac_min:
                    fac = fac_min
                elif fac > fac_max:
                    fac = fac_max

            h = h_abs * fac
            if h < h_min:
                h = h_min
            elif h > 1.0:
                h = 1.0

        xf = exp(u)
        yf = exp(v)
        if xf < 0.0:
            xf = 0.0
        if yf < 0.0:
            yf = 0.0
        return [xf, yf]

    def solve(self, problem: Any, **kwargs: Any) -> Any:
        # Batch support
        if isinstance(problem, (list, tuple)):
            return [self.solve(p, **kwargs) for p in problem]

        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = problem["y0"]
        x0 = float(y0[0])
        y0v = float(y0[1])

        params = problem["params"]
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        delta = float(params["delta"])
        gamma = float(params["gamma"])

        nb = Solver._nb_fun
        if nb is not None:
            xf, yf = nb(t0, t1, x0, y0v, alpha, beta, delta, gamma)
            return [float(xf), float(yf)]

        return self._dopri5_py(t0, t1, x0, y0v, alpha, beta, delta, gamma)