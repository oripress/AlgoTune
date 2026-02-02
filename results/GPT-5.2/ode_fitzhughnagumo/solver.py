from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

from numba import njit

@njit(cache=True, fastmath=True, inline="always")
def _fhn_rhs(v: float, w: float, ab: float, ac: float, I: float) -> tuple[float, float]:
    # dv/dt = v - v^3/3 - w + I
    dv = v - (v * v * v) * (1.0 / 3.0) - w + I
    # dw/dt = a(bv - cw) = (ab)*v - (ac)*w
    dw = ab * v - ac * w
    return dv, dw

# Dormand–Prince 5(4) (DOPRI5) coefficients (same family as SciPy RK45)
@njit(cache=True, fastmath=True)
def _integrate_fhn_dopri5(
    t0: float,
    t1: float,
    v0: float,
    w0: float,
    a: float,
    b: float,
    c: float,
    I: float,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    if t1 == t0:
        return v0, w0

    # Support both forward/backward integration without swapping initial conditions.
    direction = 1.0
    if t1 < t0:
        direction = -1.0

    # Precompute for RHS
    ab = a * b
    ac = a * c

    t = t0
    v = v0
    w = w0

    # Initial step: heuristic (kept simple for speed)
    dt_total = abs(t1 - t0)
    h = 0.1
    if dt_total < h:
        h = dt_total
    h *= direction

    # Safety parameters
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0

    # Hard caps to avoid infinite loops in pathological cases
    max_steps = 5_000_000

    # FSAL: reuse derivative at end of last accepted step as next step's k1
    k1v, k1w = _fhn_rhs(v, w, ab, ac, I)

    for _ in range(max_steps):
        remaining = t1 - t
        if direction * remaining <= 0.0:
            break

        # Don't overshoot
        if direction * h > direction * remaining:
            h = remaining

        # Stages (k1v, k1w already available)
        v2 = v + h * (0.2 * k1v)
        w2 = w + h * (0.2 * k1w)
        k2v, k2w = _fhn_rhs(v2, w2, ab, ac, I)

        v3 = v + h * (0.075 * k1v + 0.225 * k2v)
        w3 = w + h * (0.075 * k1w + 0.225 * k2w)
        k3v, k3w = _fhn_rhs(v3, w3, ab, ac, I)

        v4 = v + h * ((44.0 / 45.0) * k1v + (-56.0 / 15.0) * k2v + (32.0 / 9.0) * k3v)
        w4 = w + h * ((44.0 / 45.0) * k1w + (-56.0 / 15.0) * k2w + (32.0 / 9.0) * k3w)
        k4v, k4w = _fhn_rhs(v4, w4, ab, ac, I)

        v5 = v + h * (
            (19372.0 / 6561.0) * k1v
            + (-25360.0 / 2187.0) * k2v
            + (64448.0 / 6561.0) * k3v
            + (-212.0 / 729.0) * k4v
        )
        w5 = w + h * (
            (19372.0 / 6561.0) * k1w
            + (-25360.0 / 2187.0) * k2w
            + (64448.0 / 6561.0) * k3w
            + (-212.0 / 729.0) * k4w
        )
        k5v, k5w = _fhn_rhs(v5, w5, ab, ac, I)

        v6 = v + h * (
            (9017.0 / 3168.0) * k1v
            + (-355.0 / 33.0) * k2v
            + (46732.0 / 5247.0) * k3v
            + (49.0 / 176.0) * k4v
            + (-5103.0 / 18656.0) * k5v
        )
        w6 = w + h * (
            (9017.0 / 3168.0) * k1w
            + (-355.0 / 33.0) * k2w
            + (46732.0 / 5247.0) * k3w
            + (49.0 / 176.0) * k4w
            + (-5103.0 / 18656.0) * k5w
        )
        k6v, k6w = _fhn_rhs(v6, w6, ab, ac, I)

        # 5th order solution (b)
        v_new = v + h * (
            (35.0 / 384.0) * k1v
            + (500.0 / 1113.0) * k3v
            + (125.0 / 192.0) * k4v
            + (-2187.0 / 6784.0) * k5v
            + (11.0 / 84.0) * k6v
        )
        w_new = w + h * (
            (35.0 / 384.0) * k1w
            + (500.0 / 1113.0) * k3w
            + (125.0 / 192.0) * k4w
            + (-2187.0 / 6784.0) * k5w
            + (11.0 / 84.0) * k6w
        )

        # k7 at t+h (for error estimate b_hat uses k7)
        k7v, k7w = _fhn_rhs(v_new, w_new, ab, ac, I)

        # 4th order solution (b_hat)
        v4th = v + h * (
            (5179.0 / 57600.0) * k1v
            + (7571.0 / 16695.0) * k3v
            + (393.0 / 640.0) * k4v
            + (-92097.0 / 339200.0) * k5v
            + (187.0 / 2100.0) * k6v
            + (1.0 / 40.0) * k7v
        )
        w4th = w + h * (
            (5179.0 / 57600.0) * k1w
            + (7571.0 / 16695.0) * k3w
            + (393.0 / 640.0) * k4w
            + (-92097.0 / 339200.0) * k5w
            + (187.0 / 2100.0) * k6w
            + (1.0 / 40.0) * k7w
        )

        ev = v_new - v4th
        ew = w_new - w4th

        sv = atol + rtol * max(abs(v), abs(v_new))
        sw = atol + rtol * max(abs(w), abs(w_new))

        # Squared RMS norm over 2 components:
        # err = sqrt( ( (ev/sv)^2 + (ew/sw)^2 ) / 2 )
        e1 = ev / sv
        e2 = ew / sw
        err2 = e1 * e1 + e2 * e2  # = 2*err^2

        if err2 <= 2.0 or abs(h) <= 1e-14:
            # accept
            t += h
            v = v_new
            w = w_new
            # FSAL: derivative at the new point becomes next step's k1
            k1v = k7v
            k1w = k7w
        # Step size update (whether accepted or rejected)
        if err2 == 0.0:
            factor = max_factor
        else:
            # err**(-0.2) = (err2/2)**(-0.1)
            factor = safety * math.pow(0.5 * err2, -0.1)
            if factor < min_factor:
                factor = min_factor
            elif factor > max_factor:
                factor = max_factor

        h *= factor

    return v, w

@dataclass
class Solver:
    """Fast solver for FitzHugh–Nagumo final state using a jitted RK45 (DOPRI5)."""

    # compile once in __post_init__ (init time not counted)
    def __post_init__(self) -> None:
        _integrate_fhn_dopri5(
            0.0,
            1.0,
            -1.0,
            -0.5,
            0.08,
            0.8,
            0.7,
            0.5,
            1e-8,
            1e-8,
        )

    def solve(self, problem, **kwargs) -> Any:
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = problem["y0"]
        v0 = float(y0[0])
        w0 = float(y0[1])

        params = problem["params"]
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])

        v, w = _integrate_fhn_dopri5(t0, t1, v0, w0, a, b, c, I, 1e-8, 1e-8)
        return [float(v), float(w)]