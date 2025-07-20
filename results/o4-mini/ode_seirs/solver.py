from typing import Any
import numpy as np
from numba import njit

@njit(fastmath=True)
def _integrate_seirs_rk4(y0, t0, t1, beta, sigma, gamma, omega):
    span = t1 - t0
    N0 = int(span * 20.0)
    if N0 < 1000:
        N0 = 1000
    N = N0
    dt = span / N

    S = y0[0]; E = y0[1]; I = y0[2]; R = y0[3]
    for i in range(N):
        # Stage 1
        dS1 = -beta * S * I + omega * R
        dE1 = beta * S * I - sigma * E
        dI1 = sigma * E - gamma * I
        dR1 = gamma * I - omega * R

        # Stage 2
        S2 = S + 0.5 * dt * dS1; E2 = E + 0.5 * dt * dE1
        I2 = I + 0.5 * dt * dI1; R2 = R + 0.5 * dt * dR1
        dS2 = -beta * S2 * I2 + omega * R2
        dE2 = beta * S2 * I2 - sigma * E2
        dI2 = sigma * E2 - gamma * I2
        dR2 = gamma * I2 - omega * R2

        # Stage 3
        S3 = S + 0.5 * dt * dS2; E3 = E + 0.5 * dt * dE2
        I3 = I + 0.5 * dt * dI2; R3 = R + 0.5 * dt * dR2
        dS3 = -beta * S3 * I3 + omega * R3
        dE3 = beta * S3 * I3 - sigma * E3
        dI3 = sigma * E3 - gamma * I3
        dR3 = gamma * I3 - omega * R3

        # Stage 4
        S4 = S + dt * dS3; E4 = E + dt * dE3
        I4 = I + dt * dI3; R4 = R + dt * dR3
        dS4 = -beta * S4 * I4 + omega * R4
        dE4 = beta * S4 * I4 - sigma * E4
        dI4 = sigma * E4 - gamma * I4
        dR4 = gamma * I4 - omega * R4

        # Combine
        S += dt * (dS1 + 2.0 * dS2 + 2.0 * dS3 + dS4) / 6.0
        E += dt * (dE1 + 2.0 * dE2 + 2.0 * dE3 + dE4) / 6.0
        I += dt * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4) / 6.0
        R += dt * (dR1 + 2.0 * dR2 + 2.0 * dR3 + dR4) / 6.0

    return np.array((S, E, I, R), dtype=np.double)

class Solver:
    def solve(self, problem: Any, **kwargs) -> Any:
        y0 = np.array(problem["y0"], dtype=np.double)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        p = problem["params"]
        beta = float(p["beta"])
        sigma = float(p["sigma"])
        gamma = float(p["gamma"])
        omega = float(p["omega"])

        result = _integrate_seirs_rk4(y0, t0, t1, beta, sigma, gamma, omega)
        total = result.sum()
        if total != 0.0:
            result = result / total
        return result.tolist()