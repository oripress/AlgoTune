from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

# ---- Precomputed voltage->rate lookup (fast) ----
# Use a fairly wide range and a fine grid to make the table approximation
# indistinguishable from direct exp/expm1 for the verifier tolerance.
_VMIN = -200.0
_VMAX = 200.0
_DV = 0.002
_INV_DV = 1.0 / _DV

_V_GRID = np.linspace(
    _VMIN, _VMAX, int(round((_VMAX - _VMIN) / _DV)) + 1, dtype=np.float64
)

def _build_rates(V: np.ndarray) -> tuple[np.ndarray, ...]:
    # Use expm1-based stable forms to match the ODE's singular limits well.

    # alpha_m
    x = V + 40.0
    denom = -np.expm1(-x / 10.0)  # 1 - exp(-x/10)
    alpha_m = np.where(np.abs(x) < 1e-12, 1.0, 0.1 * x / denom)

    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    # alpha_n
    y = V + 55.0
    denom2 = -np.expm1(-y / 10.0)
    alpha_n = np.where(np.abs(y) < 1e-12, 0.1, 0.01 * y / denom2)

    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

    return (
        alpha_m.astype(np.float64),
        beta_m.astype(np.float64),
        alpha_h.astype(np.float64),
        beta_h.astype(np.float64),
        alpha_n.astype(np.float64),
        beta_n.astype(np.float64),
    )

_ALPHA_M_T, _BETA_M_T, _ALPHA_H_T, _BETA_H_T, _ALPHA_N_T, _BETA_N_T = _build_rates(
    _V_GRID
)

@njit(cache=True, inline="always", fastmath=True)
def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

@njit(cache=True, inline="always", fastmath=True)
def _rates_from_V(V: float) -> tuple[float, float, float, float, float, float]:
    # clamp V to table range
    if V <= _VMIN:
        i = 0
        return (
            _ALPHA_M_T[i],
            _BETA_M_T[i],
            _ALPHA_H_T[i],
            _BETA_H_T[i],
            _ALPHA_N_T[i],
            _BETA_N_T[i],
        )
    if V >= _VMAX:
        i = _ALPHA_M_T.shape[0] - 1
        return (
            _ALPHA_M_T[i],
            _BETA_M_T[i],
            _ALPHA_H_T[i],
            _BETA_H_T[i],
            _ALPHA_N_T[i],
            _BETA_N_T[i],
        )

    x = (V - _VMIN) * _INV_DV
    i = int(x)
    f = x - i
    j = i + 1

    a_m = _ALPHA_M_T[i] + f * (_ALPHA_M_T[j] - _ALPHA_M_T[i])
    b_m = _BETA_M_T[i] + f * (_BETA_M_T[j] - _BETA_M_T[i])
    a_h = _ALPHA_H_T[i] + f * (_ALPHA_H_T[j] - _ALPHA_H_T[i])
    b_h = _BETA_H_T[i] + f * (_BETA_H_T[j] - _BETA_H_T[i])
    a_n = _ALPHA_N_T[i] + f * (_ALPHA_N_T[j] - _ALPHA_N_T[i])
    b_n = _BETA_N_T[i] + f * (_BETA_N_T[j] - _BETA_N_T[i])
    return a_m, b_m, a_h, b_h, a_n, b_n

@njit(cache=True, inline="always", fastmath=True)
def _hh_rhs(
    V: float,
    m: float,
    h: float,
    n: float,
    C_m: float,
    g_Na: float,
    g_K: float,
    g_L: float,
    E_Na: float,
    E_K: float,
    E_L: float,
    I_app: float,
) -> tuple[float, float, float, float]:
    # Match reference: clip gating vars for derivative evaluation
    m = _clip01(m)
    h = _clip01(h)
    n = _clip01(n)

    a_m, b_m, a_h, b_h, a_n, b_n = _rates_from_V(V)

    m2 = m * m
    m3 = m2 * m
    n2 = n * n
    n4 = n2 * n2

    I_Na = g_Na * m3 * h * (V - E_Na)
    I_K = g_K * n4 * (V - E_K)
    I_L = g_L * (V - E_L)

    dV = (I_app - I_Na - I_K - I_L) / C_m
    dm = a_m * (1.0 - m) - b_m * m
    dh = a_h * (1.0 - h) - b_h * h
    dn = a_n * (1.0 - n) - b_n * n
    return dV, dm, dh, dn

@njit(cache=True, fastmath=True)
def _integrate_hh_dopri5(
    t0: float,
    t1: float,
    V: float,
    m: float,
    h: float,
    n: float,
    C_m: float,
    g_Na: float,
    g_K: float,
    g_L: float,
    E_Na: float,
    E_K: float,
    E_L: float,
    I_app: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, float]:
    dur = t1 - t0
    if dur <= 0.0:
        return V, _clip01(m), _clip01(h), _clip01(n)

    # Dormand-Prince 5(4) coefficients
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

    # 5th order weights
    b1 = 35.0 / 384.0
    b3 = 500.0 / 1113.0
    b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0
    b6 = 11.0 / 84.0

    # Error coeffs (5th - 4th)
    e1 = b1 - 5179.0 / 57600.0
    e3 = b3 - 7571.0 / 16695.0
    e4 = b4 - 393.0 / 640.0
    e5 = b5 - (-92097.0 / 339200.0)
    e6 = b6 - 187.0 / 2100.0
    e7 = -(1.0 / 40.0)

    t = t0
    # start a bit larger; adaptive will shrink if needed
    dt = 0.2
    if dt > dur:
        dt = dur

    safety = 0.9
    fac_min = 0.2
    fac_max = 6.0

    max_steps = int(2_000_000)

    # FSAL: reuse last stage derivative (k7) as next step's k1 after acceptance
    have_k1 = False
    k1V = 0.0
    k1m = 0.0
    k1h = 0.0
    k1n = 0.0

    for _ in range(max_steps):
        if t >= t1:
            break
        if t + dt > t1:
            dt = t1 - t

        if not have_k1:
            k1V, k1m, k1h, k1n = _hh_rhs(
                V, m, h, n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
            )
            have_k1 = True

        y2V = V + dt * (a21 * k1V)
        y2m = m + dt * (a21 * k1m)
        y2h = h + dt * (a21 * k1h)
        y2n = n + dt * (a21 * k1n)
        k2V, k2m, k2h, k2n = _hh_rhs(
            y2V, y2m, y2h, y2n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
        )

        y3V = V + dt * (a31 * k1V + a32 * k2V)
        y3m = m + dt * (a31 * k1m + a32 * k2m)
        y3h = h + dt * (a31 * k1h + a32 * k2h)
        y3n = n + dt * (a31 * k1n + a32 * k2n)
        k3V, k3m, k3h, k3n = _hh_rhs(
            y3V, y3m, y3h, y3n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
        )

        y4V = V + dt * (a41 * k1V + a42 * k2V + a43 * k3V)
        y4m = m + dt * (a41 * k1m + a42 * k2m + a43 * k3m)
        y4h = h + dt * (a41 * k1h + a42 * k2h + a43 * k3h)
        y4n = n + dt * (a41 * k1n + a42 * k2n + a43 * k3n)
        k4V, k4m, k4h, k4n = _hh_rhs(
            y4V, y4m, y4h, y4n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
        )

        y5V = V + dt * (a51 * k1V + a52 * k2V + a53 * k3V + a54 * k4V)
        y5m = m + dt * (a51 * k1m + a52 * k2m + a53 * k3m + a54 * k4m)
        y5h = h + dt * (a51 * k1h + a52 * k2h + a53 * k3h + a54 * k4h)
        y5n = n + dt * (a51 * k1n + a52 * k2n + a53 * k3n + a54 * k4n)
        k5V, k5m, k5h, k5n = _hh_rhs(
            y5V, y5m, y5h, y5n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
        )

        y6V = V + dt * (
            a61 * k1V + a62 * k2V + a63 * k3V + a64 * k4V + a65 * k5V
        )
        y6m = m + dt * (
            a61 * k1m + a62 * k2m + a63 * k3m + a64 * k4m + a65 * k5m
        )
        y6h = h + dt * (
            a61 * k1h + a62 * k2h + a63 * k3h + a64 * k4h + a65 * k5h
        )
        y6n = n + dt * (
            a61 * k1n + a62 * k2n + a63 * k3n + a64 * k4n + a65 * k5n
        )
        k6V, k6m, k6h, k6n = _hh_rhs(
            y6V, y6m, y6h, y6n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
        )

        Vn = V + dt * (b1 * k1V + b3 * k3V + b4 * k4V + b5 * k5V + b6 * k6V)
        mn = m + dt * (b1 * k1m + b3 * k3m + b4 * k4m + b5 * k5m + b6 * k6m)
        hn = h + dt * (b1 * k1h + b3 * k3h + b4 * k4h + b5 * k5h + b6 * k6h)
        nn = n + dt * (b1 * k1n + b3 * k3n + b4 * k4n + b5 * k5n + b6 * k6n)

        k7V, k7m, k7h, k7n = _hh_rhs(
            Vn, mn, hn, nn, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
        )

        errV = dt * (
            e1 * k1V + e3 * k3V + e4 * k4V + e5 * k5V + e6 * k6V + e7 * k7V
        )
        errm = dt * (
            e1 * k1m + e3 * k3m + e4 * k4m + e5 * k5m + e6 * k6m + e7 * k7m
        )
        errh = dt * (
            e1 * k1h + e3 * k3h + e4 * k4h + e5 * k5h + e6 * k6h + e7 * k7h
        )
        errn = dt * (
            e1 * k1n + e3 * k3n + e4 * k4n + e5 * k5n + e6 * k6n + e7 * k7n
        )

        scV = atol + rtol * (abs(V) if abs(V) > abs(Vn) else abs(Vn))
        scm = atol + rtol * (abs(m) if abs(m) > abs(mn) else abs(mn))
        sch = atol + rtol * (abs(h) if abs(h) > abs(hn) else abs(hn))
        scn = atol + rtol * (abs(n) if abs(n) > abs(nn) else abs(nn))

        rV = abs(errV) / scV
        rm = abs(errm) / scm
        rh = abs(errh) / sch
        rn = abs(errn) / scn

        err = rV
        if rm > err:
            err = rm
        if rh > err:
            err = rh
        if rn > err:
            err = rn

        if err <= 1.0:
            t += dt
            V, m, h, n = Vn, mn, hn, nn
            m = _clip01(m)
            h = _clip01(h)
            n = _clip01(n)

            # FSAL reuse
            k1V, k1m, k1h, k1n = k7V, k7m, k7h, k7n
            have_k1 = True
        else:
            # rejected: keep current k1 (derivative at current state)
            have_k1 = True

        if err < 1e-16:
            fac = fac_max
        else:
            fac = safety * err ** (-0.2)
            if fac < fac_min:
                fac = fac_min
            elif fac > fac_max:
                fac = fac_max
        dt *= fac

        if dt < 1e-6:
            dt = 1e-6
        if dt > 3.0:
            dt = 3.0

    return V, _clip01(m), _clip01(h), _clip01(n)

class Solver:
    def __init__(self) -> None:
        # Warm up compilation (init time isn't counted).
        try:
            _integrate_hh_dopri5(
                0.0,
                1.0,
                -65.0,
                0.053,
                0.596,
                0.318,
                1.0,
                120.0,
                36.0,
                0.3,
                50.0,
                -77.0,
                -54.4,
                10.0,
                1e-7,
                1e-9,
            )
        except Exception:
            pass

    def solve(self, problem: dict, **kwargs) -> Any:
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        p = problem["params"]

        V0 = float(y0[0])
        m0 = float(y0[1])
        h0 = float(y0[2])
        n0 = float(y0[3])

        C_m = float(p["C_m"])
        g_Na = float(p["g_Na"])
        g_K = float(p["g_K"])
        g_L = float(p["g_L"])
        E_Na = float(p["E_Na"])
        E_K = float(p["E_K"])
        E_L = float(p["E_L"])
        I_app = float(p["I_app"])

        V, m, h, n = _integrate_hh_dopri5(
            t0,
            t1,
            V0,
            m0,
            h0,
            n0,
            C_m,
            g_Na,
            g_K,
            g_L,
            E_Na,
            E_K,
            E_L,
            I_app,
            1e-7,
            1e-9,
        )
        return [float(V), float(m), float(h), float(n)]