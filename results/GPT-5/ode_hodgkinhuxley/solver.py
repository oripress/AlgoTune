from __future__ import annotations

from typing import Any, Dict, List
from math import exp, fabs

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the Hodgkin-Huxley ODE system and return the final state [V, m, h, n] at t1.
        Implements a fast, pure-Python Dormand–Prince RK45 (5th order with 4th order error control)
        with adaptive step size. Tolerances are matched to the reference (rtol=1e-8, atol=1e-8).
        FSAL property is used to reuse the last stage derivative for the next step.
        Direct error estimator avoids extra embedded-solution computations.
        """
        # Local aliases for builtins to reduce global lookups
        ffabs = fabs
        fmax = max
        fmin = min

        # Unpack problem
        y0_in = problem["y0"]
        yV = float(y0_in[0])
        ym = float(y0_in[1])
        yh = float(y0_in[2])
        yn = float(y0_in[3])

        t = float(problem["t0"])
        t_end = float(problem["t1"])
        params = problem["params"]

        # Pre-extract parameters (avoid dict lookups within RHS)
        C_m = float(params["C_m"])
        g_Na = float(params["g_Na"])
        g_K = float(params["g_K"])
        g_L = float(params["g_L"])
        E_Na = float(params["E_Na"])
        E_K = float(params["E_K"])
        E_L = float(params["E_L"])
        I_app = float(params["I_app"])
        inv_C_m = 1.0 / C_m

        # RHS of Hodgkin-Huxley model; optimized for scalar math
        def f_rhs(
            V: float,
            m: float,
            h: float,
            n: float,
            _gNa=g_Na,
            _gK=g_K,
            _gL=g_L,
            _ENa=E_Na,
            _EK=E_K,
            _EL=E_L,
            _I=I_app,
            _invC=inv_C_m,
            _exp=exp,
            _inv10=0.1,
            _inv18=(1.0 / 18.0),
            _inv20=(1.0 / 20.0),
            _inv80=(1.0 / 80.0),
        ):
            # Rate constants with singularity handling matching reference
            if V == -40.0:
                alpha_m = 1.0
            else:
                vp40 = V + 40.0
                alpha_m = 0.1 * vp40 / (1.0 - _exp(-vp40 * _inv10))

            v65 = V + 65.0
            beta_m = 4.0 * _exp(-v65 * _inv18)
            alpha_h = 0.07 * _exp(-v65 * _inv20)

            v35 = V + 35.0
            beta_h = 1.0 / (1.0 + _exp(-v35 * _inv10))

            if V == -55.0:
                alpha_n = 0.1
            else:
                vp55 = V + 55.0
                alpha_n = 0.01 * vp55 / (1.0 - _exp(-vp55 * _inv10))
            beta_n = 0.125 * _exp(-v65 * _inv80)

            # Clamp gating variables to [0, 1]
            if m < 0.0:
                m = 0.0
            elif m > 1.0:
                m = 1.0
            if h < 0.0:
                h = 0.0
            elif h > 1.0:
                h = 1.0
            if n < 0.0:
                n = 0.0
            elif n > 1.0:
                n = 1.0

            # Ionic currents
            m2 = m * m
            m3 = m2 * m
            n2 = n * n
            n4 = n2 * n2

            I_Na = _gNa * m3 * h * (V - _ENa)
            I_K = _gK * n4 * (V - _EK)
            I_L = _gL * (V - _EL)

            # ODEs
            dVdt = (_I - I_Na - I_K - I_L) * _invC
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n

            return dVdt, dmdt, dhdt, dndt

        # Tolerances (match reference)
        rtol = 1e-8
        atol = 1e-8

        # Dormand–Prince RK45 coefficients (floats)
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

        b1 = 35.0 / 384.0
        b3 = 500.0 / 1113.0
        b4 = 125.0 / 192.0
        b5 = -2187.0 / 6784.0
        b6 = 11.0 / 84.0

        bs1 = 5179.0 / 57600.0
        bs3 = 7571.0 / 16695.0
        bs4 = 393.0 / 640.0
        bs5 = -92097.0 / 339200.0
        bs6 = 187.0 / 2100.0
        bs7 = 1.0 / 40.0

        # Error coefficients e = b - bs (for direct error estimate)
        e1 = b1 - bs1
        e3 = b3 - bs3
        e4 = b4 - bs4
        e5 = b5 - bs5
        e6 = b6 - bs6
        e7 = -bs7

        # Adaptive control parameters
        safety = 0.9
        min_factor = 0.2
        max_factor = 20.0
        error_exponent = 1.0 / 5.0  # 5th order

        dt_total = t_end - t
        if dt_total == 0.0:
            return [yV, ym, yh, yn]

        # Initial derivative
        k1V, k1m, k1h, k1n = f_rhs(yV, ym, yh, yn)

        # Initial step selection (simple, robust)
        scaleV = atol + rtol * ffabs(yV)
        scalem = atol + rtol * ffabs(ym)
        scaleh = atol + rtol * ffabs(yh)
        scalen = atol + rtol * ffabs(yn)
        d0 = fmax(ffabs(yV) / scaleV, fmax(ffabs(ym) / scalem, fmax(ffabs(yh) / scaleh, ffabs(yn) / scalen)))
        d1 = fmax(ffabs(k1V) / scaleV, fmax(ffabs(k1m) / scalem, fmax(ffabs(k1h) / scaleh, ffabs(k1n) / scalen)))
        if d0 < 1e-5 or d1 < 1e-5:
            h = 1e-3 * dt_total
        else:
            h = 0.05 * d0 / d1  # slightly larger initial step than classic 0.01
        # Ensure direction and reasonable magnitude
        if dt_total > 0.0:
            if h <= 1e-12:
                h = 1e-12
            elif h > dt_total:
                h = dt_total
        else:
            if h >= -1e-12:
                h = -1e-12
            elif h < dt_total:
                h = dt_total

        # Main integration loop
        if dt_total > 0.0:
            while t < t_end:
                # Do not overshoot
                if t + h > t_end:
                    h = t_end - t

                # Stage 2
                y2V = yV + h * (a21 * k1V)
                y2m = ym + h * (a21 * k1m)
                y2h = yh + h * (a21 * k1h)
                y2n = yn + h * (a21 * k1n)
                k2V, k2m, k2h, k2n = f_rhs(y2V, y2m, y2h, y2n)

                # Stage 3
                y3V = yV + h * (a31 * k1V + a32 * k2V)
                y3m = ym + h * (a31 * k1m + a32 * k2m)
                y3h = yh + h * (a31 * k1h + a32 * k2h)
                y3n = yn + h * (a31 * k1n + a32 * k2n)
                k3V, k3m, k3h, k3n = f_rhs(y3V, y3m, y3h, y3n)

                # Stage 4
                y4V = yV + h * (a41 * k1V + a42 * k2V + a43 * k3V)
                y4m = ym + h * (a41 * k1m + a42 * k2m + a43 * k3m)
                y4h = yh + h * (a41 * k1h + a42 * k2h + a43 * k3h)
                y4n = yn + h * (a41 * k1n + a42 * k2n + a43 * k3n)
                k4V, k4m, k4h, k4n = f_rhs(y4V, y4m, y4h, y4n)

                # Stage 5
                y5V_ = yV + h * (a51 * k1V + a52 * k2V + a53 * k3V + a54 * k4V)
                y5m_ = ym + h * (a51 * k1m + a52 * k2m + a53 * k3m + a54 * k4m)
                y5h_ = yh + h * (a51 * k1h + a52 * k2h + a53 * k3h + a54 * k4h)
                y5n_ = yn + h * (a51 * k1n + a52 * k2n + a53 * k3n + a54 * k4n)
                k5V, k5m, k5h, k5n = f_rhs(y5V_, y5m_, y5h_, y5n_)

                # Stage 6
                y6V = yV + h * (a61 * k1V + a62 * k2V + a63 * k3V + a64 * k4V + a65 * k5V)
                y6m = ym + h * (a61 * k1m + a62 * k2m + a63 * k3m + a64 * k4m + a65 * k5m)
                y6h = yh + h * (a61 * k1h + a62 * k2h + a63 * k3h + a64 * k4h + a65 * k5h)
                y6n = yn + h * (a61 * k1n + a62 * k2n + a63 * k3n + a64 * k4n + a65 * k5n)
                k6V, k6m, k6h, k6n = f_rhs(y6V, y6m, y6h, y6n)

                # 5th order solution (use b-coefficients)
                y5V = yV + h * (b1 * k1V + b3 * k3V + b4 * k4V + b5 * k5V + b6 * k6V)
                y5m = ym + h * (b1 * k1m + b3 * k3m + b4 * k4m + b5 * k5m + b6 * k6m)
                y5h = yh + h * (b1 * k1h + b3 * k3h + b4 * k4h + b5 * k5h + b6 * k6h)
                y5n = yn + h * (b1 * k1n + b3 * k3n + b4 * k4n + b5 * k5n + b6 * k6n)

                # Stage 7 (FSAL derivative at y5)
                k7V, k7m, k7h, k7n = f_rhs(y5V, y5m, y5h, y5n)

                # Direct error estimate using e = b - bs
                h_abs = ffabs(h)
                errV = h_abs * ffabs(e1 * k1V + e3 * k3V + e4 * k4V + e5 * k5V + e6 * k6V + e7 * k7V) / (
                    atol + rtol * fmax(ffabs(yV), ffabs(y5V))
                )
                errm = h_abs * ffabs(e1 * k1m + e3 * k3m + e4 * k4m + e5 * k5m + e6 * k6m + e7 * k7m) / (
                    atol + rtol * fmax(ffabs(ym), ffabs(y5m))
                )
                errh = h_abs * ffabs(e1 * k1h + e3 * k3h + e4 * k4h + e5 * k5h + e6 * k6h + e7 * k7h) / (
                    atol + rtol * fmax(ffabs(yh), ffabs(y5h))
                )
                errn = h_abs * ffabs(e1 * k1n + e3 * k3n + e4 * k4n + e5 * k5n + e6 * k6n + e7 * k7n) / (
                    atol + rtol * fmax(ffabs(yn), ffabs(y5n))
                )
                err_norm = fmax(errV, fmax(errm, fmax(errh, errn)))

                if err_norm <= 1.0:
                    # Accept step
                    t += h
                    yV, ym, yh, yn = y5V, y5m, y5h, y5n

                    # FSAL: reuse last stage derivative as k1 for next step
                    k1V, k1m, k1h, k1n = k7V, k7m, k7h, k7n

                    # Update step size
                    if err_norm == 0.0:
                        factor = max_factor
                    else:
                        factor = fmin(max_factor, fmax(min_factor, safety * (err_norm ** (-error_exponent))))
                    h *= factor
                else:
                    # Reject step, decrease h (keep same k1 at this point)
                    factor = fmax(min_factor, safety * (err_norm ** (-error_exponent)))
                    h *= factor

                # Avoid infinitesimal steps
                if ffabs(h) < 1e-16:
                    break
        else:
            while t > t_end:
                # Do not overshoot
                if t + h < t_end:
                    h = t_end - t

                # Stage 2
                y2V = yV + h * (a21 * k1V)
                y2m = ym + h * (a21 * k1m)
                y2h = yh + h * (a21 * k1h)
                y2n = yn + h * (a21 * k1n)
                k2V, k2m, k2h, k2n = f_rhs(y2V, y2m, y2h, y2n)

                # Stage 3
                y3V = yV + h * (a31 * k1V + a32 * k2V)
                y3m = ym + h * (a31 * k1m + a32 * k2m)
                y3h = yh + h * (a31 * k1h + a32 * k2h)
                y3n = yn + h * (a31 * k1n + a32 * k2n)
                k3V, k3m, k3h, k3n = f_rhs(y3V, y3m, y3h, y3n)

                # Stage 4
                y4V = yV + h * (a41 * k1V + a42 * k2V + a43 * k3V)
                y4m = ym + h * (a41 * k1m + a42 * k2m + a43 * k3m)
                y4h = yh + h * (a41 * k1h + a42 * k2h + a43 * k3h)
                y4n = yn + h * (a41 * k1n + a42 * k2n + a43 * k3n)
                k4V, k4m, k4h, k4n = f_rhs(y4V, y4m, y4h, y4n)

                # Stage 5
                y5V_ = yV + h * (a51 * k1V + a52 * k2V + a53 * k3V + a54 * k4V)
                y5m_ = ym + h * (a51 * k1m + a52 * k2m + a53 * k3m + a54 * k4m)
                y5h_ = yh + h * (a51 * k1h + a52 * k2h + a53 * k3h + a54 * k4h)
                y5n_ = yn + h * (a51 * k1n + a52 * k2n + a53 * k3n + a54 * k4n)
                k5V, k5m, k5h, k5n = f_rhs(y5V_, y5m_, y5h_, y5n_)

                # Stage 6
                y6V = yV + h * (a61 * k1V + a62 * k2V + a63 * k3V + a64 * k4V + a65 * k5V)
                y6m = ym + h * (a61 * k1m + a62 * k2m + a63 * k3m + a64 * k4m + a65 * k5m)
                y6h = yh + h * (a61 * k1h + a62 * k2h + a63 * k3h + a64 * k4h + a65 * k5h)
                y6n = yn + h * (a61 * k1n + a62 * k2n + a63 * k3n + a64 * k4n + a65 * k5n)
                k6V, k6m, k6h, k6n = f_rhs(y6V, y6m, y6h, y6n)

                # 5th order solution (use b-coefficients)
                y5V = yV + h * (b1 * k1V + b3 * k3V + b4 * k4V + b5 * k5V + b6 * k6V)
                y5m = ym + h * (b1 * k1m + b3 * k3m + b4 * k4m + b5 * k5m + b6 * k6m)
                y5h = yh + h * (b1 * k1h + b3 * k3h + b4 * k4h + b5 * k5h + b6 * k6h)
                y5n = yn + h * (b1 * k1n + b3 * k3n + b4 * k4n + b5 * k5n + b6 * k6n)

                # Stage 7 (FSAL derivative at y5)
                k7V, k7m, k7h, k7n = f_rhs(y5V, y5m, y5h, y5n)

                # Direct error estimate using e = b - bs
                h_abs = ffabs(h)
                errV = h_abs * ffabs(e1 * k1V + e3 * k3V + e4 * k4V + e5 * k5V + e6 * k6V + e7 * k7V) / (
                    atol + rtol * fmax(ffabs(yV), ffabs(y5V))
                )
                errm = h_abs * ffabs(e1 * k1m + e3 * k3m + e4 * k4m + e5 * k5m + e6 * k6m + e7 * k7m) / (
                    atol + rtol * fmax(ffabs(ym), ffabs(y5m))
                )
                errh = h_abs * ffabs(e1 * k1h + e3 * k3h + e4 * k4h + e5 * k5h + e6 * k6h + e7 * k7h) / (
                    atol + rtol * fmax(ffabs(yh), ffabs(y5h))
                )
                errn = h_abs * ffabs(e1 * k1n + e3 * k3n + e4 * k4n + e5 * k5n + e6 * k6n + e7 * k7n) / (
                    atol + rtol * fmax(ffabs(yn), ffabs(y5n))
                )
                err_norm = fmax(errV, fmax(errm, fmax(errh, errn)))

                if err_norm <= 1.0:
                    # Accept step
                    t += h
                    yV, ym, yh, yn = y5V, y5m, y5h, y5n

                    # FSAL: reuse last stage derivative as k1 for next step
                    k1V, k1m, k1h, k1n = k7V, k7m, k7h, k7n

                    # Update step size
                    if err_norm == 0.0:
                        factor = max_factor
                    else:
                        factor = fmin(max_factor, fmax(min_factor, safety * (err_norm ** (-error_exponent))))
                    h *= factor
                else:
                    # Reject step, decrease h (keep same k1 at this point)
                    factor = fmax(min_factor, safety * (err_norm ** (-error_exponent)))
                    h *= factor

                # Avoid infinitesimal steps
                if ffabs(h) < 1e-16:
                    break

        return [yV, ym, yh, yn]