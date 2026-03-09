from __future__ import annotations

from math import exp, fabs, sqrt
from typing import Any

from numba import njit

@njit(cache=True)
def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

@njit(cache=True)
def _vtrap(z: float) -> float:
    if -1e-7 < z < 1e-7:
        zz = z * z
        return 1.0 + 0.5 * z + zz / 12.0 - zz * zz / 720.0
    return z / (1.0 - exp(-z))

@njit(cache=True)
def _rhs(
    V: float,
    m: float,
    h: float,
    n: float,
    Cm: float,
    gNa: float,
    gK: float,
    gL: float,
    ENa: float,
    EK: float,
    EL: float,
    Iapp: float,
) -> tuple[float, float, float, float]:
    mc = _clip01(m)
    hc = _clip01(h)
    nc = _clip01(n)

    am = _vtrap((V + 40.0) / 10.0)
    bm = 4.0 * exp(-(V + 65.0) / 18.0)

    ah = 0.07 * exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))

    an = 0.1 * _vtrap((V + 55.0) / 10.0)
    bn = 0.125 * exp(-(V + 65.0) / 80.0)

    m2 = mc * mc
    n2 = nc * nc

    INa = gNa * m2 * mc * hc * (V - ENa)
    IK = gK * n2 * n2 * (V - EK)
    IL = gL * (V - EL)

    dV = (Iapp - INa - IK - IL) / Cm
    dm = am * (1.0 - mc) - bm * mc
    dh = ah * (1.0 - hc) - bh * hc
    dn = an * (1.0 - nc) - bn * nc
    return dV, dm, dh, dn

@njit(cache=True)
def _rms4(a0: float, a1: float, a2: float, a3: float) -> float:
    return sqrt((a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3) * 0.25)

@njit(cache=True)
def _select_initial_step(
    t0: float,
    t1: float,
    V: float,
    m: float,
    h: float,
    n: float,
    Cm: float,
    gNa: float,
    gK: float,
    gL: float,
    ENa: float,
    EK: float,
    EL: float,
    Iapp: float,
    rtol: float,
    atol: float,
) -> float:
    fV, fm, fh, fn = _rhs(V, m, h, n, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

    sV = atol + rtol * fabs(V)
    sm = atol + rtol * fabs(m)
    sh = atol + rtol * fabs(h)
    sn = atol + rtol * fabs(n)

    d0 = _rms4(V / sV, m / sm, h / sh, n / sn)
    d1 = _rms4(fV / sV, fm / sm, fh / sh, fn / sn)

    dt = fabs(t1 - t0)
    if dt == 0.0:
        return 0.0

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    if h0 > dt:
        h0 = dt

    V1 = V + h0 * fV
    m1 = m + h0 * fm
    h1s = h + h0 * fh
    n1 = n + h0 * fn

    fV1, fm1, fh1, fn1 = _rhs(V1, m1, h1s, n1, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

    d2 = _rms4(
        (fV1 - fV) / sV,
        (fm1 - fm) / sm,
        (fh1 - fh) / sh,
        (fn1 - fn) / sn,
    ) / h0

    if d1 < 1e-15 and d2 < 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2

    h = min(100.0 * h0, h1, dt)
    if h < 1e-8:
        h = min(1e-8, dt)
    return h

@njit(cache=True)
def _solve_numba(
    t0: float,
    t1: float,
    V: float,
    m: float,
    h: float,
    n: float,
    Cm: float,
    gNa: float,
    gK: float,
    gL: float,
    ENa: float,
    EK: float,
    EL: float,
    Iapp: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, float]:
    dt_total = t1 - t0
    if dt_total == 0.0:
        return V, _clip01(m), _clip01(h), _clip01(n)

    direction = 1.0
    if dt_total < 0.0:
        direction = -1.0

    step = _select_initial_step(
        t0, t1, V, m, h, n, Cm, gNa, gK, gL, ENa, EK, EL, Iapp, rtol, atol
    )
    if step == 0.0:
        return V, _clip01(m), _clip01(h), _clip01(n)
    hstep = direction * step

    t = t0
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0

    k1V, k1m, k1h, k1n = _rhs(V, m, h, n, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)
    rejected = False
    count = 0

    while direction * (t1 - t) > 0.0:
        count += 1
        if count > 2000000:
            break

        if direction * (t + hstep - t1) > 0.0:
            hstep = t1 - t

        V2 = V + hstep * (0.2 * k1V)
        m2 = m + hstep * (0.2 * k1m)
        h2 = h + hstep * (0.2 * k1h)
        n2 = n + hstep * (0.2 * k1n)
        k2V, k2m, k2h, k2n = _rhs(V2, m2, h2, n2, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

        V3 = V + hstep * (0.075 * k1V + 0.225 * k2V)
        m3 = m + hstep * (0.075 * k1m + 0.225 * k2m)
        h3 = h + hstep * (0.075 * k1h + 0.225 * k2h)
        n3 = n + hstep * (0.075 * k1n + 0.225 * k2n)
        k3V, k3m, k3h, k3n = _rhs(V3, m3, h3, n3, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

        V4 = V + hstep * (
            (44.0 / 45.0) * k1V + (-56.0 / 15.0) * k2V + (32.0 / 9.0) * k3V
        )
        m4 = m + hstep * (
            (44.0 / 45.0) * k1m + (-56.0 / 15.0) * k2m + (32.0 / 9.0) * k3m
        )
        h4 = h + hstep * (
            (44.0 / 45.0) * k1h + (-56.0 / 15.0) * k2h + (32.0 / 9.0) * k3h
        )
        n4 = n + hstep * (
            (44.0 / 45.0) * k1n + (-56.0 / 15.0) * k2n + (32.0 / 9.0) * k3n
        )
        k4V, k4m, k4h, k4n = _rhs(V4, m4, h4, n4, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

        V5 = V + hstep * (
            (19372.0 / 6561.0) * k1V
            + (-25360.0 / 2187.0) * k2V
            + (64448.0 / 6561.0) * k3V
            + (-212.0 / 729.0) * k4V
        )
        m5 = m + hstep * (
            (19372.0 / 6561.0) * k1m
            + (-25360.0 / 2187.0) * k2m
            + (64448.0 / 6561.0) * k3m
            + (-212.0 / 729.0) * k4m
        )
        h5 = h + hstep * (
            (19372.0 / 6561.0) * k1h
            + (-25360.0 / 2187.0) * k2h
            + (64448.0 / 6561.0) * k3h
            + (-212.0 / 729.0) * k4h
        )
        n5 = n + hstep * (
            (19372.0 / 6561.0) * k1n
            + (-25360.0 / 2187.0) * k2n
            + (64448.0 / 6561.0) * k3n
            + (-212.0 / 729.0) * k4n
        )
        k5V, k5m, k5h, k5n = _rhs(V5, m5, h5, n5, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

        V6 = V + hstep * (
            (9017.0 / 3168.0) * k1V
            + (-355.0 / 33.0) * k2V
            + (46732.0 / 5247.0) * k3V
            + (49.0 / 176.0) * k4V
            + (-5103.0 / 18656.0) * k5V
        )
        m6 = m + hstep * (
            (9017.0 / 3168.0) * k1m
            + (-355.0 / 33.0) * k2m
            + (46732.0 / 5247.0) * k3m
            + (49.0 / 176.0) * k4m
            + (-5103.0 / 18656.0) * k5m
        )
        h6 = h + hstep * (
            (9017.0 / 3168.0) * k1h
            + (-355.0 / 33.0) * k2h
            + (46732.0 / 5247.0) * k3h
            + (49.0 / 176.0) * k4h
            + (-5103.0 / 18656.0) * k5h
        )
        n6 = n + hstep * (
            (9017.0 / 3168.0) * k1n
            + (-355.0 / 33.0) * k2n
            + (46732.0 / 5247.0) * k3n
            + (49.0 / 176.0) * k4n
            + (-5103.0 / 18656.0) * k5n
        )
        k6V, k6m, k6h, k6n = _rhs(V6, m6, h6, n6, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

        Vn = V + hstep * (
            (35.0 / 384.0) * k1V
            + (500.0 / 1113.0) * k3V
            + (125.0 / 192.0) * k4V
            + (-2187.0 / 6784.0) * k5V
            + (11.0 / 84.0) * k6V
        )
        mn = m + hstep * (
            (35.0 / 384.0) * k1m
            + (500.0 / 1113.0) * k3m
            + (125.0 / 192.0) * k4m
            + (-2187.0 / 6784.0) * k5m
            + (11.0 / 84.0) * k6m
        )
        hn = h + hstep * (
            (35.0 / 384.0) * k1h
            + (500.0 / 1113.0) * k3h
            + (125.0 / 192.0) * k4h
            + (-2187.0 / 6784.0) * k5h
            + (11.0 / 84.0) * k6h
        )
        nn = n + hstep * (
            (35.0 / 384.0) * k1n
            + (500.0 / 1113.0) * k3n
            + (125.0 / 192.0) * k4n
            + (-2187.0 / 6784.0) * k5n
            + (11.0 / 84.0) * k6n
        )

        k7V, k7m, k7h, k7n = _rhs(Vn, mn, hn, nn, Cm, gNa, gK, gL, ENa, EK, EL, Iapp)

        eV = hstep * (
            (71.0 / 57600.0) * k1V
            + (-71.0 / 16695.0) * k3V
            + (71.0 / 1920.0) * k4V
            + (-17253.0 / 339200.0) * k5V
            + (22.0 / 525.0) * k6V
            + (-1.0 / 40.0) * k7V
        )
        em = hstep * (
            (71.0 / 57600.0) * k1m
            + (-71.0 / 16695.0) * k3m
            + (71.0 / 1920.0) * k4m
            + (-17253.0 / 339200.0) * k5m
            + (22.0 / 525.0) * k6m
            + (-1.0 / 40.0) * k7m
        )
        eh = hstep * (
            (71.0 / 57600.0) * k1h
            + (-71.0 / 16695.0) * k3h
            + (71.0 / 1920.0) * k4h
            + (-17253.0 / 339200.0) * k5h
            + (22.0 / 525.0) * k6h
            + (-1.0 / 40.0) * k7h
        )
        en = hstep * (
            (71.0 / 57600.0) * k1n
            + (-71.0 / 16695.0) * k3n
            + (71.0 / 1920.0) * k4n
            + (-17253.0 / 339200.0) * k5n
            + (22.0 / 525.0) * k6n
            + (-1.0 / 40.0) * k7n
        )

        sV = atol + rtol * max(fabs(V), fabs(Vn))
        sm = atol + rtol * max(fabs(m), fabs(mn))
        sh = atol + rtol * max(fabs(h), fabs(hn))
        sn = atol + rtol * max(fabs(n), fabs(nn))

        err = sqrt(
            (
                (eV / sV) * (eV / sV)
                + (em / sm) * (em / sm)
                + (eh / sh) * (eh / sh)
                + (en / sn) * (en / sn)
            )
            * 0.25
        )

        if err <= 1.0:
            t += hstep
            V = Vn
            m = mn
            h = hn
            n = nn
            k1V, k1m, k1h, k1n = k7V, k7m, k7h, k7n

            if err == 0.0:
                factor = max_factor
            else:
                factor = safety * err ** (-0.2)
                if factor < min_factor:
                    factor = min_factor
                elif factor > max_factor:
                    factor = max_factor

            if rejected and factor > 1.0:
                factor = 1.0
            hstep *= factor
            rejected = False
        else:
            factor = safety * err ** (-0.2)
            if factor < min_factor:
                factor = min_factor
            elif factor > 1.0:
                factor = 1.0
            hstep *= factor
            rejected = True

        if fabs(hstep) < 1e-12:
            hstep = direction * 1e-12

    return V, _clip01(m), _clip01(h), _clip01(n)

class Solver:
    def __init__(self) -> None:
        _solve_numba(
            0.0,
            0.1,
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
            1e-8,
            1e-8,
        )

    def solve(self, problem, **kwargs) -> Any:
        p = problem["params"]
        V, m, h, n = _solve_numba(
            float(problem["t0"]),
            float(problem["t1"]),
            float(problem["y0"][0]),
            float(problem["y0"][1]),
            float(problem["y0"][2]),
            float(problem["y0"][3]),
            float(p["C_m"]),
            float(p["g_Na"]),
            float(p["g_K"]),
            float(p["g_L"]),
            float(p["E_Na"]),
            float(p["E_K"]),
            float(p["E_L"]),
            float(p["I_app"]),
            1e-8,
            1e-8,
        )
        return [V, m, h, n]