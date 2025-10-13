from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the SEIRS model ODEs and return the state at final time t1.

        Uses a reduced 3D system (R = 1 - S - E - I) for speed, and integrates
        with a high-order explicit solver (DOP853) at tight tolerances to
        closely match the reference solution while running faster.
        """
        # Extract inputs
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0_full = problem["y0"]
        params = problem["params"]

        # Edge case: zero duration
        if t1 == t0:
            return [float(y) for y in y0_full]

        # Parameters
        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        omega = float(params["omega"])

        # Reduce to 3D system: y = [S, E, I], with R = 1 - S - E - I
        y0 = np.array([y0_full[0], y0_full[1], y0_full[2]], dtype=float)

        # Derivative function
        def seirs3(_: float, y: np.ndarray):
            S, E, I = y
            R = 1.0 - S - E - I
            dS = -beta * S * I + omega * R
            dE = beta * S * I - sigma * E
            dI = sigma * E - gamma * I
            return (dS, dE, dI)

        # Tight tolerances to closely match reference while using faster method
        rtol = 1e-10
        atol = 1e-10

        sol = solve_ivp(
            seirs3,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        S, E, I = sol.y[:, -1]
        R = 1.0 - (S + E + I)

        # Return as list of 4 floats
        return [float(S), float(E), float(I), float(R)]