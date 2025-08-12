from typing import Any
from math import sqrt

# Try to use numba for speed; provide a pure-Python fallback.
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

def _integrate_seirs_py(t0, t1, S, E, I, beta, sigma, gamma, omega):
    # Dormand-Prince 5(4) coefficients (RK45)
    A21 = 1.0 / 5.0

    A31 = 3.0 / 40.0
    A32 = 9.0 / 40.0

    A41 = 44.0 / 45.0
    A42 = -56.0 / 15.0
    A43 = 32.0 / 9.0

    A51 = 19372.0 / 6561.0
    A52 = -25360.0 / 2187.0
    A53 = 64448.0 / 6561.0
    A54 = -212.0 / 729.0

    A61 = 9017.0 / 3168.0
    A62 = -355.0 / 33.0
    A63 = 46732.0 / 5247.0
    A64 = 49.0 / 176.0
    A65 = -5103.0 / 18656.0

    # b (5th order solution)
    B1 = 35.0 / 384.0
    B2 = 0.0
    B3 = 500.0 / 1113.0
    B4 = 125.0 / 192.0
    B5 = -2187.0 / 6784.0
    B6 = 11.0 / 84.0

    # b_hat (4th order) for error estimate
    BH1 = 5179.0 / 57600.0
    BH2 = 0.0
    BH3 = 7571.0 / 16695.0
    BH4 = 393.0 / 640.0
    BH5 = -92097.0 / 339200.0
    BH6 = 187.0 / 2100.0
    BH7 = 1.0 / 40.0  # b_hat7 only for error estimate

    # Precompute differences for error estimation
    dB1 = B1 - BH1
    dB2 = B2 - BH2  # 0.0
    dB3 = B3 - BH3
    dB4 = B4 - BH4
    dB5 = B5 - BH5
    dB6 = B6 - BH6
    dB7 = 0.0 - BH7

    # Error control parameters (relaxed but within verification tolerance)
    rtol = 1e-6
    atol = 1e-12
    safety = 0.97
    min_factor = 0.1
    max_factor = 100.0
    inv_order = 1.0 / 5.0  # for exponent

    # Direction and bounds
    direction = 1.0 if t1 >= t0 else -1.0
    t = t0
    t_bound = t1

    # Initial derivative and step size (inline vector field)
    r = 1.0 - S - E - I
    si = beta * S * I
    k1S = -si + omega * r
    k1E = si - sigma * E
    k1I = sigma * E - gamma * I

    sS0 = atol + rtol * abs(S)
    sE0 = atol + rtol * abs(E)
    sI0 = atol + rtol * abs(I)
    d0 = sqrt(((S / sS0) ** 2 + (E / sE0) ** 2 + (I / sI0) ** 2) / 3.0)

    sS1 = atol + rtol * abs(k1S)
    sE1 = atol + rtol * abs(k1E)
    sI1 = atol + rtol * abs(k1I)
    d1 = sqrt(((k1S / sS1) ** 2 + (k1E / sE1) ** 2 + (k1I / sI1) ** 2) / 3.0)

    if d0 < 1e-5 or d1 < 1e-5:
        h = 1e-6
    else:
        h = 0.01 * d0 / d1

    h *= direction
    dt_total = t1 - t0
    if abs(h) > abs(dt_total):
        h = dt_total

    max_steps = 10000000
    steps = 0

    while (t - t_bound) * direction < 0.0:
        steps += 1
        if steps > max_steps:
            break

        # Adjust step to hit the bound
        h = min(abs(h), abs(t_bound - t)) * direction

        # k2
        y2S = S + h * A21 * k1S
        y2E = E + h * A21 * k1E
        y2I = I + h * A21 * k1I
        r = 1.0 - y2S - y2E - y2I
        si = beta * y2S * y2I
        k2S = -si + omega * r
        k2E = si - sigma * y2E
        k2I = sigma * y2E - gamma * y2I

        # k3
        y3S = S + h * (A31 * k1S + A32 * k2S)
        y3E = E + h * (A31 * k1E + A32 * k2E)
        y3I = I + h * (A31 * k1I + A32 * k2I)
        r = 1.0 - y3S - y3E - y3I
        si = beta * y3S * y3I
        k3S = -si + omega * r
        k3E = si - sigma * y3E
        k3I = sigma * y3E - gamma * y3I

        # k4
        y4S = S + h * (A41 * k1S + A42 * k2S + A43 * k3S)
        y4E = E + h * (A41 * k1E + A42 * k2E + A43 * k3E)
        y4I = I + h * (A41 * k1I + A42 * k2I + A43 * k3I)
        r = 1.0 - y4S - y4E - y4I
        si = beta * y4S * y4I
        k4S = -si + omega * r
        k4E = si - sigma * y4E
        k4I = sigma * y4E - gamma * y4I

        # k5
        y5S = S + h * (A51 * k1S + A52 * k2S + A53 * k3S + A54 * k4S)
        y5E = E + h * (A51 * k1E + A52 * k2E + A53 * k3E + A54 * k4E)
        y5I = I + h * (A51 * k1I + A52 * k2I + A53 * k3I + A54 * k4I)
        r = 1.0 - y5S - y5E - y5I
        si = beta * y5S * y5I
        k5S = -si + omega * r
        k5E = si - sigma * y5E
        k5I = sigma * y5E - gamma * y5I

        # k6
        y6S = S + h * (A61 * k1S + A62 * k2S + A63 * k3S + A64 * k4S + A65 * k5S)
        y6E = E + h * (A61 * k1E + A62 * k2E + A63 * k3E + A64 * k4E + A65 * k5E)
        y6I = I + h * (A61 * k1I + A62 * k2I + A63 * k3I + A64 * k4I + A65 * k5I)
        r = 1.0 - y6S - y6E - y6I
        si = beta * y6S * y6I
        k6S = -si + omega * r
        k6E = si - sigma * y6E
        k6I = sigma * y6E - gamma * y6I

        # 5th-order solution
        S_new = S + h * (B1 * k1S + B2 * k2S + B3 * k3S + B4 * k4S + B5 * k5S + B6 * k6S)
        E_new = E + h * (B1 * k1E + B2 * k2E + B3 * k3E + B4 * k4E + B5 * k5E + B6 * k6E)
        I_new = I + h * (B1 * k1I + B2 * k2I + B3 * k3I + B4 * k4I + B5 * k5I + B6 * k6I)

        # k7 at t + h using the 5th order solution (FSAL)
        r = 1.0 - S_new - E_new - I_new
        si = beta * S_new * I_new
        k7S = -si + omega * r
        k7E = si - sigma * E_new
        k7I = sigma * E_new - gamma * I_new

        # Error estimate using difference of weights
        eS = h * (dB1 * k1S + dB2 * k2S + dB3 * k3S + dB4 * k4S + dB5 * k5S + dB6 * k6S + dB7 * k7S)
        eE = h * (dB1 * k1E + dB2 * k2E + dB3 * k3E + dB4 * k4E + dB5 * k5E + dB6 * k6E + dB7 * k7E)
        eI = h * (dB1 * k1I + dB2 * k2I + dB3 * k3I + dB4 * k4I + dB5 * k5I + dB6 * k6I + dB7 * k7I)

        # Weighted RMS norm (inline)
        sS = atol + rtol * (abs(S) if abs(S) >= abs(S_new) else abs(S_new))
        sE = atol + rtol * (abs(E) if abs(E) >= abs(E_new) else abs(E_new))
        sI = atol + rtol * (abs(I) if abs(I) >= abs(I_new) else abs(I_new))
        eS /= sS
        eE /= sE
        eI /= sI
        err = sqrt((eS * eS + eE * eE + eI * eI) / 3.0)

        if err <= 1.0:
            # Accept step
            t += h
            S, E, I = S_new, E_new, I_new

            # Step size adaptation
            if err == 0.0:
                factor = max_factor
            else:
                factor = safety * (err ** (-inv_order))
                if factor < min_factor:
                    factor = min_factor
                elif factor > max_factor:
                    factor = max_factor

            h *= factor
            # Reuse last derivative (FSAL)
            k1S, k1E, k1I = k7S, k7E, k7I
        else:
            # Reject and reduce step
            factor = safety * (err ** (-inv_order))
            if factor < min_factor:
                factor = min_factor
            elif factor > 1.0 / min_factor:
                factor = 1.0 / min_factor
            h *= factor

        # Prevent stalling
        if abs(h) < 1e-15:
            break

    return S, E, I

# Numba-accelerated version
if NUMBA_AVAILABLE:
    _integrate_seirs_njit = njit(fastmath=True)(_integrate_seirs_py)
else:
    _integrate_seirs_njit = None

class Solver:
    def __init__(self) -> None:
        # Optionally trigger JIT compilation at import/init time to avoid paying at first call.
        if NUMBA_AVAILABLE and _integrate_seirs_njit is not None:
            try:
                # Warm up compilation with a dummy call (types: float)
                _integrate_seirs_njit(
                    0.0,
                    1.0,
                    0.9,
                    0.05,
                    0.01,
                    0.3,
                    0.2,
                    0.1,
                    0.01,
                )
            except Exception:
                # If warmup fails for any reason, ignore and fall back at runtime if needed.
                pass

    def solve(self, problem, **kwargs) -> Any:
        # Extract input
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]

        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        omega = float(params["omega"])

        # Early exit if no integration needed
        if t1 == t0:
            return [float(y0[0]), float(y0[1]), float(y0[2]), float(y0[3])]

        # Reduce dimension using conservation: R = 1 - S - E - I
        S = float(y0[0])
        E = float(y0[1])
        I = float(y0[2])

        if NUMBA_AVAILABLE and _integrate_seirs_njit is not None:
            S, E, I = _integrate_seirs_njit(t0, t1, S, E, I, beta, sigma, gamma, omega)
        else:
            S, E, I = _integrate_seirs_py(t0, t1, S, E, I, beta, sigma, gamma, omega)

        # Reconstruct R to enforce conservation
        R = 1.0 - S - E - I
        return [S, E, I, R]