from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    # fallback: define a no-op decorator
    def njit(signature_or_function=None, **kwargs):
        def wrapper(func):
            return func
        if callable(signature_or_function):
            return signature_or_function
        return wrapper

# ---------------------------
# Numba-accelerated components
# ---------------------------

@njit(cache=True, fastmath=False)
def _alpha_beta_rates(V: float):
    # Rate constants with handling of singularity points
    # alpha_m
    d = V + 40.0
    if abs(d) < 1e-12:
        alpha_m = 1.0  # L'Hôpital limit at V = -40 mV
    else:
        alpha_m = 0.1 * d / (1.0 - np.exp(-d / 10.0))

    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    d2 = V + 55.0
    if abs(d2) < 1e-12:
        alpha_n = 0.1  # L'Hôpital limit at V = -55 mV
    else:
        alpha_n = 0.01 * d2 / (1.0 - np.exp(-d2 / 10.0))

    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

@njit(cache=True, fastmath=False)
def _hh_rhs(
    t: float,
    y: np.ndarray,
    C_m: float,
    g_Na: float,
    g_K: float,
    g_L: float,
    E_Na: float,
    E_K: float,
    E_L: float,
    I_app: float,
    out: np.ndarray,
) -> None:
    # Unpack and clip gating variables to [0,1] as in the reference
    V = y[0]
    m = y[1]
    h = y[2]
    n = y[3]

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

    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = _alpha_beta_rates(V)

    # Ionic currents
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)

    # ODEs
    out[0] = (I_app - I_Na - I_K - I_L) / C_m
    out[1] = alpha_m * (1.0 - m) - beta_m * m
    out[2] = alpha_h * (1.0 - h) - beta_h * h
    out[3] = alpha_n * (1.0 - n) - beta_n * n

@njit(cache=True, fastmath=False)
def _weighted_rms_norm(err: np.ndarray, y: np.ndarray, y_new: np.ndarray, atol: float, rtol: float) -> float:
    s0 = 0.0
    for i in range(err.shape[0]):
        scale = atol + rtol * (abs(y_new[i]))  # use y_new scaling
        val = err[i] / scale
        s0 += val * val
    return np.sqrt(s0 / err.shape[0])

@njit(cache=True, fastmath=False)
def _rk45_integrate(
    t0: float,
    t1: float,
    y0: np.ndarray,
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
    max_steps: int = 10000000,
) -> np.ndarray:
    # Dormand–Prince (RK45) coefficients
    # c
    c2 = 1.0 / 5.0
    c3 = 3.0 / 10.0
    c4 = 4.0 / 5.0
    c5 = 8.0 / 9.0
    c6 = 1.0
    c7 = 1.0

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

    # b for 5th-order solution
    b1 = 35.0 / 384.0
    b2 = 0.0
    b3 = 500.0 / 1113.0
    b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0
    b6 = 11.0 / 84.0
    b7 = 0.0

    # b_hat for 4th-order solution
    bh1 = 5179.0 / 57600.0
    bh2 = 0.0
    bh3 = 7571.0 / 16695.0
    bh4 = 393.0 / 640.0
    bh5 = -92097.0 / 339200.0
    bh6 = 187.0 / 2100.0
    bh7 = 1.0 / 40.0

    # Initialization
    t = t0
    y = y0.copy()

    f = np.empty_like(y0)
    _hh_rhs(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, f)

    # Initial step size heuristic
    # Based on Hairer/Wanner style
    # Compute scaling
    ssum = 0.0
    for i in range(y.shape[0]):
        scale = atol + rtol * abs(y[i])
        val = y[i] / scale
        ssum += val * val
    d0 = np.sqrt(ssum / y.shape[0])

    ssum = 0.0
    for i in range(y.shape[0]):
        scale = atol + rtol * abs(y[i])
        val = f[i] / scale
        ssum += val * val
    d1 = np.sqrt(ssum / y.shape[0])

    h0 = 1e-6
    if d0 >= 1e-5 and d1 > 1e-5:
        h0 = 0.01 * d0 / d1

    # Limit initial step to not overshoot and be positive
    h = min(max(h0, 1e-8), max(1e-8, (t1 - t0) / 10.0))
    if h <= 0.0:
        h = 1e-6

    # Preallocate stage arrays
    k1 = np.empty_like(y)
    k2 = np.empty_like(y)
    k3 = np.empty_like(y)
    k4 = np.empty_like(y)
    k5 = np.empty_like(y)
    k6 = np.empty_like(y)
    k7 = np.empty_like(y)

    y_temp = np.empty_like(y)
    y5 = np.empty_like(y)
    y4 = np.empty_like(y)
    err = np.empty_like(y)

    # Main integration loop
    n_steps = 0
    while t < t1 and n_steps < max_steps:
        if t + h > t1:
            h = t1 - t

        # k1
        for i in range(y.shape[0]):
            k1[i] = f[i]

        # k2
        for i in range(y.shape[0]):
            y_temp[i] = y[i] + h * (a21 * k1[i])
        _hh_rhs(t + c2 * h, y_temp, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, k2)

        # k3
        for i in range(y.shape[0]):
            y_temp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i])
        _hh_rhs(t + c3 * h, y_temp, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, k3)

        # k4
        for i in range(y.shape[0]):
            y_temp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
        _hh_rhs(t + c4 * h, y_temp, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, k4)

        # k5
        for i in range(y.shape[0]):
            y_temp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
        _hh_rhs(t + c5 * h, y_temp, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, k5)

        # k6
        for i in range(y.shape[0]):
            y_temp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
        _hh_rhs(t + c6 * h, y_temp, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, k6)

        # k7 (for dense output pair)
        for i in range(y.shape[0]):
            y_temp[i] = y[i] + h * (a71 * k1[i] + 0.0 * k2[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i])
        _hh_rhs(t + c7 * h, y_temp, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, k7)

        # 5th-order solution
        for i in range(y.shape[0]):
            y5[i] = y[i] + h * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i] + b7 * k7[i])
        # 4th-order solution
        for i in range(y.shape[0]):
            y4[i] = y[i] + h * (bh1 * k1[i] + bh2 * k2[i] + bh3 * k3[i] + bh4 * k4[i] + bh5 * k5[i] + bh6 * k6[i] + bh7 * k7[i])

        # error
        for i in range(y.shape[0]):
            err[i] = y5[i] - y4[i]

        err_norm = _weighted_rms_norm(err, y, y5, atol, rtol)

        # step adapt
        if err_norm <= 1.0:
            # accept
            t = t + h
            for i in range(y.shape[0]):
                y[i] = y5[i]

            # clip gating variables to [0,1]
            if y[1] < 0.0:
                y[1] = 0.0
            elif y[1] > 1.0:
                y[1] = 1.0
            if y[2] < 0.0:
                y[2] = 0.0
            elif y[2] > 1.0:
                y[2] = 1.0
            if y[3] < 0.0:
                y[3] = 0.0
            elif y[3] > 1.0:
                y[3] = 1.0

            # compute f at new step for next iteration
            _hh_rhs(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, f)

            # new step size
            if err_norm == 0.0:
                factor = 5.0
            else:
                factor = 0.9 * err_norm ** (-0.2)  # -1/(order+1) with order=5
                if factor < 0.2:
                    factor = 0.2
                elif factor > 5.0:
                    factor = 5.0
            h = h * factor
            # ensure not too small
            if h < 1e-12:
                h = 1e-12

            n_steps += 1
        else:
            # reject
            factor = max(0.2, 0.9 * err_norm ** (-0.2))
            h = h * factor
            if h < 1e-12:
                h = 1e-12
            # do not increment step counter on rejection

    return y

class Solver:
    def __init__(self) -> None:
        # Warm-up JIT compilation so it doesn't count towards runtime
        if NUMBA_AVAILABLE:
            # Sample parameters similar to defaults
            C_m = 1.0
            g_Na = 120.0
            g_K = 36.0
            g_L = 0.3
            E_Na = 50.0
            E_K = -77.0
            E_L = -54.4
            I_app = 10.0
            t0 = 0.0
            t1 = 0.1
            y0 = np.array([-65.0, 0.053, 0.596, 0.318], dtype=np.float64)
            atol = 1e-8
            rtol = 1e-8
            # Call helper functions to ensure they are compiled
            tmp = np.empty_like(y0)
            _hh_rhs(t0, y0, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, tmp)
            _rk45_integrate(t0, t1, y0, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, rtol, atol)

    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        # Parse inputs
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]

        C_m = float(params["C_m"])
        g_Na = float(params["g_Na"])
        g_K = float(params["g_K"])
        g_L = float(params["g_L"])
        E_Na = float(params["E_Na"])
        E_K = float(params["E_K"])
        E_L = float(params["E_L"])
        I_app = float(params["I_app"])

        # Solver tolerances: match reference to ensure accuracy
        rtol = 1e-8
        atol = 1e-8

        y_final = _rk45_integrate(
            t0,
            t1,
            y0,
            C_m,
            g_Na,
            g_K,
            g_L,
            E_Na,
            E_K,
            E_L,
            I_app,
            rtol,
            atol,
        )
        # Return as list
        return y_final.tolist()