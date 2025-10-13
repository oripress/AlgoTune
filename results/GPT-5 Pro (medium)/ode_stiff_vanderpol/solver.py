from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        # Extract parameters
        mu = float(problem["mu"])
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])

        # Vector field for Van der Pol (scaled as in reference)
        # dx/dt = v
        # dv/dt = mu * ((1 - x^2) * v - x)
        def f(_t: float, y: np.ndarray) -> np.ndarray:
            x, v = y
            return np.array([v, mu * ((1.0 - x * x) * v - x)], dtype=float)

        # Analytic Jacobian (for Radau/BDF fallback)
        # J = [[0, 1],
        #      [mu*(-2*x*v - 1), mu*(1 - x^2)]]
        def jac(_t: float, y: np.ndarray) -> np.ndarray:
            x, v = y
            return np.array(
                [[0.0, 1.0], [mu * (-2.0 * x * v - 1.0), mu * (1.0 - x * x)]],
                dtype=float,
            )

        # Try LSODA first (fast robust ODEPACK with automatic stiffness detection)
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="LSODA",
            rtol=1e-8,
            atol=1e-9,
            t_eval=None,
            dense_output=False,
        )

        if not sol.success:
            # Fallback to Radau with analytic Jacobian
            sol = solve_ivp(
                f,
                (t0, t1),
                y0,
                method="Radau",
                jac=jac,
                rtol=1e-8,
                atol=1e-9,
                t_eval=None,
                dense_output=False,
            )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return final state as list
        return sol.y[:, -1].tolist()