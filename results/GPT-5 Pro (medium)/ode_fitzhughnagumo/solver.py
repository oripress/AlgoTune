from __future__ import annotations

from typing import Any, Dict, List
from math import sqrt, fabs

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        # Extract problem data
        t0: float = float(problem["t0"])
        t1: float = float(problem["t1"])
        y0_list = problem["y0"]
        v = float(y0_list[0])
        w = float(y0_list[1])

        params = problem["params"]
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])

        # Tolerances comparable to reference
        rtol = 1e-8
        atol = 1e-8

        # Weighted RMS norm for error control
        inv_sqrt2 = 1.0 / sqrt(2.0)

        def err_norm(e_v: float, e_w: float, y_v: float, y_w: float) -> float:
            sv = atol + rtol * fabs(y_v)
            sw = atol + rtol * fabs(y_w)
            return sqrt((e_v / sv) * (e_v / sv) + (e_w / sw) * (e_w / sw)) * inv_sqrt2

        # Initial step selection (lightweight heuristic, inlined rhs)
        def select_initial_step(t0_: float, v0: float, w0: float) -> float:
            dv0 = v0 - (v0 * v0 * v0) / 3.0 - w0 + I
            dw0 = a * (b * v0 - c * w0)
            sv = atol + rtol * fabs(v0)
            sw = atol + rtol * fabs(w0)
            d0 = sqrt((v0 / sv) * (v0 / sv) + (w0 / sw) * (w0 / sw)) * inv_sqrt2
            d1 = sqrt((dv0 / sv) * (dv0 / sv) + (dw0 / sw) * (dw0 / sw)) * inv_sqrt2
            if d0 < 1e-5 or d1 < 1e-5:
                h0 = 1e-3
            else:
                h0 = 0.01 * d0 / d1
            # Avoid too large initial step relative to interval
            interval = fabs(t1 - t0_)
            h0 = min(h0, interval if interval > 0 else 1.0)
            if h0 <= 1e-12:
                h0 = 1e-6
            return h0

        # Dormand-Prince (RK45) coefficients (same as scipy default method)
        # c
        c2 = 1.0 / 5.0
        c3 = 3.0 / 10.0
        c4 = 4.0 / 5.0
        c5 = 8.0 / 9.0
        # c6 = 1.0
        # c7 = 1.0

        # a
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

        a71 = 35.0 / 384.0
        # a72 = 0.0
        a73 = 500.0 / 1113.0
        a74 = 125.0 / 192.0
        a75 = -2187.0 / 6784.0
        a76 = 11.0 / 84.0

        # b (5th order solution)
        b1 = 35.0 / 384.0
        # b2 = 0.0
        b3 = 500.0 / 1113.0
        b4 = 125.0 / 192.0
        b5 = -2187.0 / 6784.0
        b6 = 11.0 / 84.0
        # b7 = 0.0

        # bs (4th order) not needed explicitly; use e = b - bs for error
        bs1 = 5179.0 / 57600.0
        # bs2 = 0.0
        bs3 = 7571.0 / 16695.0
        bs4 = 393.0 / 640.0
        bs5 = -92097.0 / 339200.0
        bs6 = 187.0 / 2100.0
        bs7 = 1.0 / 40.0

        e1 = b1 - bs1
        # e2 = 0.0
        e3 = b3 - bs3
        e4 = b4 - bs4
        e5 = b5 - bs5
        e6 = b6 - bs6
        e7 = 0.0 - bs7

        # Integration loop
        t = t0
        direction = 1.0 if t1 >= t0 else -1.0
        h = select_initial_step(t0, v, w) * direction

        h_abs_min = 1e-12

        # FSAL: reuse last stage derivative as k1 of next step if previous accepted
        have_k1 = False
        k1v = 0.0
        k1w = 0.0

        while (t - t1) * direction < 0.0:
            # Clamp last step to hit t1 exactly
            h = min(fabs(h), fabs(t1 - t)) * direction

            # Stage k1
            if not have_k1:
                # f(v, w)
                v2 = v * v
                k1v = v - (v2 * v) / 3.0 - w + I
                k1w = a * (b * v - c * w)

            # Stage k2
            vv = v + h * a21 * k1v
            ww = w + h * a21 * k1w
            v2 = vv * vv
            k2v = vv - (v2 * vv) / 3.0 - ww + I
            k2w = a * (b * vv - c * ww)

            # Stage k3
            vv = v + h * (a31 * k1v + a32 * k2v)
            ww = w + h * (a31 * k1w + a32 * k2w)
            v2 = vv * vv
            k3v = vv - (v2 * vv) / 3.0 - ww + I
            k3w = a * (b * vv - c * ww)

            # Stage k4
            vv = v + h * (a41 * k1v + a42 * k2v + a43 * k3v)
            ww = w + h * (a41 * k1w + a42 * k2w + a43 * k3w)
            v2 = vv * vv
            k4v = vv - (v2 * vv) / 3.0 - ww + I
            k4w = a * (b * vv - c * ww)

            # Stage k5
            vv = v + h * (a51 * k1v + a52 * k2v + a53 * k3v + a54 * k4v)
            ww = w + h * (a51 * k1w + a52 * k2w + a53 * k3w + a54 * k4w)
            v2 = vv * vv
            k5v = vv - (v2 * vv) / 3.0 - ww + I
            k5w = a * (b * vv - c * ww)

            # Stage k6
            vv = v + h * (a61 * k1v + a62 * k2v + a63 * k3v + a64 * k4v + a65 * k5v)
            ww = w + h * (a61 * k1w + a62 * k2w + a63 * k3w + a64 * k4w + a65 * k5w)
            v2 = vv * vv
            k6v = vv - (v2 * vv) / 3.0 - ww + I
            k6w = a * (b * vv - c * ww)

            # Stage k7 (at t+h, y+h*sum(a7i*ki)), FSAL
            vv = v + h * (a71 * k1v + a73 * k3v + a74 * k4v + a75 * k5v + a76 * k6v)
            ww = w + h * (a71 * k1w + a73 * k3w + a74 * k4w + a75 * k5w + a76 * k6w)
            v2 = vv * vv
            k7v = vv - (v2 * vv) / 3.0 - ww + I
            k7w = a * (b * vv - c * ww)

            # 5th order solution y5
            y5v = v + h * (b1 * k1v + b3 * k3v + b4 * k4v + b5 * k5v + b6 * k6v)
            y5w = w + h * (b1 * k1w + b3 * k3w + b4 * k4w + b5 * k5w + b6 * k6w)

            # Error estimate: ev = h * sum_i ( (b_i - bs_i) * k_i )
            ev = h * (e1 * k1v + e3 * k3v + e4 * k4v + e5 * k5v + e6 * k6v + e7 * k7v)
            ew = h * (e1 * k1w + e3 * k3w + e4 * k4w + e5 * k5w + e6 * k6w + e7 * k7w)
            err = err_norm(ev, ew, y5v, y5w)

            if err <= 1.0:
                # Accept step
                t += h
                v, w = y5v, y5w

                # FSAL: next k1 is current k7
                k1v, k1w = k7v, k7w
                have_k1 = True

                # Step size control (order = 5)
                if err == 0.0:
                    factor = 10.0
                else:
                    factor = 0.9 * (1.0 / err) ** 0.2
                    if factor > 10.0:
                        factor = 10.0
                    elif factor < 0.2:
                        factor = 0.2
                h *= factor
            else:
                # Reject step: decrease h, reuse existing k1 (since start state unchanged)
                factor = 0.9 * (1.0 / err) ** 0.2
                if factor < 0.2:
                    factor = 0.2
                new_h = h * factor
                if fabs(new_h) < h_abs_min:
                    # If step size underflows, accept to prevent infinite loop
                    t += h
                    v, w = y5v, y5w
                    k1v, k1w = k7v, k7w
                    have_k1 = True
                    h = h_abs_min * direction
                else:
                    h = new_h
                    # have_k1 stays True if it was True; k1 still equals f(v,w)

        return [float(v), float(w)]