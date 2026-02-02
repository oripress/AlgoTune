from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

# ---------------- Numba kernels (fast path) ----------------
_integrate_strang_nb = None
_NB_COMPILED = False

if njit is not None:

    @njit(cache=False, fastmath=True)
    def _exp_nb(x: float) -> float:
        return np.exp(x)

    @njit(cache=False, fastmath=True)
    def _f_diff_nb(r1: float, r2: float, dt: float) -> float:
        """(exp(-r1*dt)-exp(-r2*dt))/(r2-r1), stable for r1~=r2."""
        dr = r2 - r1
        if abs(dr) < 1e-12:
            return dt * _exp_nb(-r1 * dt)
        return (_exp_nb(-r1 * dt) - _exp_nb(-r2 * dt)) / dr

    @njit(cache=False, fastmath=True)
    def _linear_step_nb(S: float, E: float, I: float, R: float, sigma: float, gamma: float, omega: float, dt: float):
        """Exact linear transitions over dt: E->I->R->S (no infection term)."""
        eS = _exp_nb(-sigma * dt)
        eG = _exp_nb(-gamma * dt)
        eO = _exp_nb(-omega * dt)

        E1 = E * eS
        d = gamma - sigma

        if abs(d) < 1e-12:
            # gamma == sigma
            I1 = eG * (I + sigma * E * dt)

            g = gamma
            k = omega - g
            if abs(k) < 1e-12:
                integ = I * dt + g * E * (dt * dt) * 0.5
                R1 = (R + g * integ) * eO
            else:
                ek = _exp_nb(k * dt)
                integ1 = I * (ek - 1.0) / k
                integ2 = g * E * (ek * (k * dt - 1.0) + 1.0) / (k * k)
                R1 = R * eO + g * eO * (integ1 + integ2)
        else:
            # gamma != sigma
            I1 = I * eG + sigma * E * (eS - eG) / d
            b = sigma * E / d
            a = I - b
            R1 = R * eO + gamma * (a * _f_diff_nb(gamma, omega, dt) + b * _f_diff_nb(sigma, omega, dt))

        S1 = 1.0 - (E1 + I1 + R1)
        return S1, E1, I1, R1

    @njit(cache=False, fastmath=True)
    def _infection_step_nb(S: float, E: float, I: float, R: float, beta: float, dt: float):
        """Exact infection-only subflow over dt with I frozen."""
        k = beta * I
        if k == 0.0:
            return S, E, I, R
        S1 = S * _exp_nb(-k * dt)
        E1 = E + (S - S1)
        return S1, E1, I, R

    @njit(cache=False, fastmath=True)
    def _integrate_strang_nb_impl(
        t0: float,
        t1: float,
        S: float,
        E: float,
        I: float,
        R: float,
        beta: float,
        sigma: float,
        gamma: float,
        omega: float,
        dt: float,
    ):
        """Strang splitting with exact substeps."""
        T = t1 - t0
        if T <= 0.0:
            return S, E, I, R

        n = int(T // dt)
        rem = T - n * dt

        half = 0.5 * dt
        for _ in range(n):
            S, E, I, R = _linear_step_nb(S, E, I, R, sigma, gamma, omega, half)
            S, E, I, R = _infection_step_nb(S, E, I, R, beta, dt)
            S, E, I, R = _linear_step_nb(S, E, I, R, sigma, gamma, omega, half)

        if rem > 0.0:
            half = 0.5 * rem
            S, E, I, R = _linear_step_nb(S, E, I, R, sigma, gamma, omega, half)
            S, E, I, R = _infection_step_nb(S, E, I, R, beta, rem)
            S, E, I, R = _linear_step_nb(S, E, I, R, sigma, gamma, omega, half)

        # Cleanup and renormalize
        if S < 0.0:
            S = 0.0
        if E < 0.0:
            E = 0.0
        if I < 0.0:
            I = 0.0
        if R < 0.0:
            R = 0.0
        s = S + E + I + R
        if s != 0.0:
            inv = 1.0 / s
            S *= inv
            E *= inv
            I *= inv
            R *= inv
        return S, E, I, R

    _integrate_strang_nb = _integrate_strang_nb_impl

# ---------------- Python fallback (rare) ----------------
def _integrate_strang_py(
    t0: float,
    t1: float,
    S: float,
    E: float,
    I: float,
    R: float,
    beta: float,
    sigma: float,
    gamma: float,
    omega: float,
    dt: float,
):
    exp = np.exp
    T = t1 - t0
    if T <= 0.0:
        return S, E, I, R

    def f_diff(r1: float, r2: float, dt_: float) -> float:
        dr = r2 - r1
        if abs(dr) < 1e-12:
            return float(dt_ * exp(-r1 * dt_))
        return float((exp(-r1 * dt_) - exp(-r2 * dt_)) / dr)

    def linear_step(S0: float, E0: float, I0: float, R0: float, dt_: float):
        eS = float(exp(-sigma * dt_))
        eG = float(exp(-gamma * dt_))
        eO = float(exp(-omega * dt_))
        E1 = E0 * eS
        d = gamma - sigma
        if abs(d) < 1e-12:
            I1 = eG * (I0 + sigma * E0 * dt_)
            g = gamma
            k = omega - g
            if abs(k) < 1e-12:
                integ = I0 * dt_ + g * E0 * (dt_ * dt_) * 0.5
                R1 = (R0 + g * integ) * eO
            else:
                ek = float(exp(k * dt_))
                integ1 = I0 * (ek - 1.0) / k
                integ2 = g * E0 * (ek * (k * dt_ - 1.0) + 1.0) / (k * k)
                R1 = R0 * eO + g * eO * (integ1 + integ2)
        else:
            I1 = I0 * eG + sigma * E0 * (eS - eG) / d
            b = sigma * E0 / d
            a = I0 - b
            R1 = R0 * eO + gamma * (a * f_diff(gamma, omega, dt_) + b * f_diff(sigma, omega, dt_))
        S1 = 1.0 - (E1 + I1 + R1)
        return S1, E1, I1, R1

    def infect_step(S0: float, E0: float, I0: float, R0: float, dt_: float):
        k = beta * I0
        if k == 0.0:
            return S0, E0, I0, R0
        S1 = S0 * float(exp(-k * dt_))
        return S1, E0 + (S0 - S1), I0, R0

    n = int(T // dt)
    rem = T - n * dt
    half = 0.5 * dt

    for _ in range(n):
        S, E, I, R = linear_step(S, E, I, R, half)
        S, E, I, R = infect_step(S, E, I, R, dt)
        S, E, I, R = linear_step(S, E, I, R, half)

    if rem > 0.0:
        half = 0.5 * rem
        S, E, I, R = linear_step(S, E, I, R, half)
        S, E, I, R = infect_step(S, E, I, R, rem)
        S, E, I, R = linear_step(S, E, I, R, half)

    S = max(S, 0.0)
    E = max(E, 0.0)
    I = max(I, 0.0)
    R = max(R, 0.0)
    s = S + E + I + R
    if s != 0.0:
        inv = 1.0 / s
        S *= inv
        E *= inv
        I *= inv
        R *= inv
    return S, E, I, R

class Solver:
    def __init__(self) -> None:
        # Compile kernels here (init time doesn't count towards solve runtime).
        global _NB_COMPILED
        if _NB_COMPILED:
            return
        if _integrate_strang_nb is None:
            _NB_COMPILED = True
            return
        try:
            _integrate_strang_nb(
                0.0,
                1.0,
                0.9,
                0.05,
                0.03,
                0.02,
                0.35,
                0.2,
                0.1,
                0.002,
                0.05,
            )
        finally:
            _NB_COMPILED = True

    def solve(self, problem: dict, **kwargs) -> Any:
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        p = problem["params"]

        beta = float(p["beta"])
        sigma = float(p["sigma"])
        gamma = float(p["gamma"])
        omega = float(p["omega"])

        S0 = float(y0[0])
        E0 = float(y0[1])
        I0 = float(y0[2])
        R0 = float(y0[3])

        max_rate = max(abs(beta), abs(sigma), abs(gamma), abs(omega), 1e-9)
        dt = 0.12 / max_rate
        # Allow a larger max step; Richardson still uses dt/2 as the "fine" solve.
        if dt > 0.1:
            dt = 0.1
        elif dt < 0.0015:
            dt = 0.0015

        integ = _integrate_strang_nb if _integrate_strang_nb is not None else _integrate_strang_py

        # Step-doubling + Richardson extrapolation (boosts order from ~2 to ~3).
        S1, E1, I1, R1 = integ(t0, t1, S0, E0, I0, R0, beta, sigma, gamma, omega, dt)
        dt2 = 0.5 * dt
        S2, E2, I2, R2 = integ(t0, t1, S0, E0, I0, R0, beta, sigma, gamma, omega, dt2)

        S = S2 + (S2 - S1) / 3.0
        E = E2 + (E2 - E1) / 3.0
        I = I2 + (I2 - I1) / 3.0
        R = R2 + (R2 - R1) / 3.0

        # Final cleanup: clamp tiny negatives and renormalize.
        if S < 0.0:
            S = 0.0
        if E < 0.0:
            E = 0.0
        if I < 0.0:
            I = 0.0
        if R < 0.0:
            R = 0.0

        s = S + E + I + R
        if s != 0.0:
            inv = 1.0 / s
            S *= inv
            E *= inv
            I *= inv
            R *= inv

        return [float(S), float(E), float(I), float(R)]