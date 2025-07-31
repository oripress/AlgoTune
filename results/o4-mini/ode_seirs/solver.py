from typing import Any
import numpy as np
from numba import njit
import math

@njit(fastmath=True)
def integrate_seirs3(S, E, I, t0, t1, beta, sigma, gamma, omega):
    dt_max = 0.2
    T = t1 - t0
    if T <= 0.0:
        return S, E, I
    N = int(math.ceil(T / dt_max))
    if N < 1:
        N = 1
    dt = T / N
    for _ in range(N):
        R = 1.0 - S - E - I
        # Stage 1
        dS1 = -beta * S * I + omega * R
        dE1 = beta * S * I - sigma * E
        dI1 = sigma * E - gamma * I
        S1 = S + 0.5 * dt * dS1
        E1 = E + 0.5 * dt * dE1
        I1 = I + 0.5 * dt * dI1
        # Stage 2
        R1 = 1.0 - S1 - E1 - I1
        dS2 = -beta * S1 * I1 + omega * R1
        dE2 = beta * S1 * I1 - sigma * E1
        dI2 = sigma * E1 - gamma * I1
        S2 = S + 0.5 * dt * dS2
        E2 = E + 0.5 * dt * dE2
        I2 = I + 0.5 * dt * dI2
        # Stage 3
        R2 = 1.0 - S2 - E2 - I2
        dS3 = -beta * S2 * I2 + omega * R2
        dE3 = beta * S2 * I2 - sigma * E2
        dI3 = sigma * E2 - gamma * I2
        S3 = S + dt * dS3
        E3 = E + dt * dE3
        I3 = I + dt * dI3
        # Stage 4
        R3 = 1.0 - S3 - E3 - I3
        dS4 = -beta * S3 * I3 + omega * R3
        dE4 = beta * S3 * I3 - sigma * E3
        dI4 = sigma * E3 - gamma * I3
        # Combine
        S += dt * (dS1 + 2 * dS2 + 2 * dS3 + dS4) / 6.0
        E += dt * (dE1 + 2 * dE2 + 2 * dE3 + dE4) / 6.0
        I += dt * (dI1 + 2 * dI2 + 2 * dI3 + dI4) / 6.0
    return S, E, I

class Solver:
    def __init__(self):
        # Warm up compilation
        self.integrator = integrate_seirs3
        self.integrator(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        # Unpack initial state and parameters
        S0, E0, I0, _ = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        p = problem["params"]
        beta = float(p["beta"])
        sigma = float(p["sigma"])
        gamma = float(p["gamma"])
        omega = float(p["omega"])
        # Integrate
        S, E, I = self.integrator(S0, E0, I0, t0, t1, beta, sigma, gamma, omega)
        R = 1.0 - S - E - I
        return [S, E, I, R]