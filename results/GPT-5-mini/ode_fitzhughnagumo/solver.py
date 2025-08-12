from typing import Any, Tuple
import math

def _dopri5_integrate(
    t0: float,
    t1: float,
    y0: Tuple[float, float],
    a: float,
    b: float,
    c: float,
    I: float,
    rtol: float,
    atol: float,
) -> Tuple[float, float]:
    """
    Dormand-Prince 5(4) adaptive integrator specialized for the 2D
    FitzHugh-Nagumo system. Returns (v, w) at t1.
    """
    # Dormand-Prince coefficients
    a21 = 1.0 / 5.0
    a31, a32 = 3.0 / 40.0, 9.0 / 40.0
    a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    a51, a52, a53, a54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
    a61, a62, a63, a64, a65 = 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0
    a71, a72, a73, a74, a75, a76 = 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0

    b1, b2, b3, b4, b5, b6, b7 = (
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    )
    bs1, bs2, bs3, bs4, bs5, bs6, bs7 = (
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    )

    # Adaptivity parameters
    safety = 0.9
    min_factor, max_factor = 0.2, 5.0
    order = 5.0
    max_iters = 20000000

    v = float(y0[0])
    w = float(y0[1])
    t = float(t0)
    total_dt = float(t1) - float(t0)
    if total_dt == 0.0:
        return (v, w)
    sign = 1.0 if total_dt > 0.0 else -1.0

    # Initial step guess
    h = sign * min(abs(total_dt), max(1e-6, min(1.0, abs(total_dt) * 0.01)))
    if h == 0.0:
        h = sign * 1e-6

    iters = 0
    while iters < max_iters:
        iters += 1

        # don't step past final time
        if (t + h - t1) * sign > 0.0:
            h = t1 - t

        # Stage 1
        vv = v * v
        vvv = vv * v
        k1_v = v - (vvv) / 3.0 - w + I
        k1_w = a * (b * v - c * w)

        # Stage 2
        v2 = v + h * (a21 * k1_v)
        w2 = w + h * (a21 * k1_w)
        vv = v2 * v2
        vvv = vv * v2
        k2_v = v2 - (vvv) / 3.0 - w2 + I
        k2_w = a * (b * v2 - c * w2)

        # Stage 3
        v3 = v + h * (a31 * k1_v + a32 * k2_v)
        w3 = w + h * (a31 * k1_w + a32 * k2_w)
        vv = v3 * v3
        vvv = vv * v3
        k3_v = v3 - (vvv) / 3.0 - w3 + I
        k3_w = a * (b * v3 - c * w3)

        # Stage 4
        v4 = v + h * (a41 * k1_v + a42 * k2_v + a43 * k3_v)
        w4 = w + h * (a41 * k1_w + a42 * k2_w + a43 * k3_w)
        vv = v4 * v4
        vvv = vv * v4
        k4_v = v4 - (vvv) / 3.0 - w4 + I
        k4_w = a * (b * v4 - c * w4)

        # Stage 5
        v5 = v + h * (a51 * k1_v + a52 * k2_v + a53 * k3_v + a54 * k4_v)
        w5 = w + h * (a51 * k1_w + a52 * k2_w + a53 * k3_w + a54 * k4_w)
        vv = v5 * v5
        vvv = vv * v5
        k5_v = v5 - (vvv) / 3.0 - w5 + I
        k5_w = a * (b * v5 - c * w5)

        # Stage 6
        v6 = v + h * (a61 * k1_v + a62 * k2_v + a63 * k3_v + a64 * k4_v + a65 * k5_v)
        w6 = w + h * (a61 * k1_w + a62 * k2_w + a63 * k3_w + a64 * k4_w + a65 * k5_w)
        vv = v6 * v6
        vvv = vv * v6
        k6_v = v6 - (vvv) / 3.0 - w6 + I
        k6_w = a * (b * v6 - c * w6)

        # Stage 7 (for final 5th order)
        v7 = v + h * (
            a71 * k1_v + a72 * k2_v + a73 * k3_v + a74 * k4_v + a75 * k5_v + a76 * k6_v
        )
        w7 = w + h * (
            a71 * k1_w + a72 * k2_w + a73 * k3_w + a74 * k4_w + a75 * k5_w + a76 * k6_w
        )
        vv = v7 * v7
        vvv = vv * v7
        k7_v = v7 - (vvv) / 3.0 - w7 + I
        k7_w = a * (b * v7 - c * w7)

        # 5th and 4th order estimates
        y5_v = v + h * (
            b1 * k1_v + b2 * k2_v + b3 * k3_v + b4 * k4_v + b5 * k5_v + b6 * k6_v + b7 * k7_v
        )
        y5_w = w + h * (
            b1 * k1_w + b2 * k2_w + b3 * k3_w + b4 * k4_w + b5 * k5_w + b6 * k6_w + b7 * k7_w
        )

        y4_v = v + h * (
            bs1 * k1_v + bs2 * k2_v + bs3 * k3_v + bs4 * k4_v + bs5 * k5_v + bs6 * k6_v + bs7 * k7_v
        )
        y4_w = w + h * (
            bs1 * k1_w + bs2 * k2_w + bs3 * k3_w + bs4 * k4_w + bs5 * k5_w + bs6 * k6_w + bs7 * k7_w
        )

        # Error estimate (RMS over two components scaled by tolerances)
        err_v = y5_v - y4_v
        err_w = y5_w - y4_w
        sc_v = atol + rtol * max(abs(v), abs(y5_v))
        sc_w = atol + rtol * max(abs(w), abs(y5_w))
        if sc_v == 0.0:
            sc_v = atol
        if sc_w == 0.0:
            sc_w = atol

        err0 = err_v / sc_v
        err1 = err_w / sc_w
        err_norm = math.sqrt((err0 * err0 + err1 * err1) / 2.0)

        if not math.isfinite(err_norm):
            # numerical trouble: reduce step
            h *= 0.1
            if abs(h) < 1e-20:
                return (v, w)
            continue

        if err_norm <= 1.0:
            # Accept step
            t += h
            v = y5_v
            w = y5_w
            # Check termination
            if (t1 - t) * sign <= 0.0 or abs(t1 - t) <= 1e-14 * max(1.0, abs(t1)):
                return (v, w)
            # Propose new step
            if err_norm == 0.0:
                factor = max_factor
            else:
                factor = safety * (err_norm ** (-1.0 / order))
                if factor < min_factor:
                    factor = min_factor
                elif factor > max_factor:
                    factor = max_factor
            h = h * factor
            if abs(h) < 1e-20:
                return (v, w)
        else:
            # Reject step, shrink
            factor = safety * (err_norm ** (-1.0 / order))
            if factor < min_factor:
                factor = min_factor
            elif factor > max_factor:
                factor = max_factor
            h = h * factor
            if abs(h) < 1e-20:
                return (v, w)

    # Fallback: return last state
    return (v, w)

class Solver:
    def __init__(self) -> None:
        pass

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solve the FitzHugh-Nagumo system from t0 to t1 and return [v, w] at t1.
        Expected problem keys: t0, t1, y0, params (a,b,c,I)
        """
        t0 = float(problem.get("t0", 0.0))
        t1 = float(problem.get("t1", 0.0))

        y0 = problem.get("y0", [0.0, 0.0])
        try:
            y0v = float(y0[0])
            y0w = float(y0[1])
        except Exception:
            if isinstance(y0, (int, float)):
                y0v = float(y0)
                y0w = 0.0
            else:
                y0v, y0w = 0.0, 0.0

        params = problem.get("params", {}) or {}
        a = float(params.get("a", 0.08))
        b = float(params.get("b", 0.8))
        c = float(params.get("c", 0.7))
        I = float(params.get("I", 0.5))

        rtol = float(kwargs.get("rtol", 1e-8))
        atol = float(kwargs.get("atol", 1e-8))

        v_final, w_final = _dopri5_integrate(t0, t1, (y0v, y0w), a, b, c, I, rtol, atol)
        return [float(v_final), float(w_final)]