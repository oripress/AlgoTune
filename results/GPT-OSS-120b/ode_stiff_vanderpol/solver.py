import numpy as np
from typing import Any, Dict, List

# SciPy is available in the execution environment; we use it for a robust stiff ODE solver.
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the stiff Van der Pol oscillator from t0 to t1.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
            - mu : float, stiffness parameter
            - y0 : list[float], initial state [x, v]
            - t0 : float, initial time
            - t1 : float, final time

        Returns
        -------
        list[float]
            Approximate state [x(t1), v(t1)].
        """
        # Extract parameters
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        mu = float(problem["mu"])

        # Define the Van der Pol system
        def vdp(t, y):
            x, v = y
            dx_dt = v
            dv_dt = mu * ((1 - x ** 2) * v - x)
            return np.array([dx_dt, dv_dt])

        # Use a stiff-aware method; Radau is a good default.
        # Tolerances are set to meet the required accuracy while remaining fast.
        def vdp_jac(t, y):
            x, v = y
            return np.array(
                [
                    [0.0, 1.0],
                    [mu * (-2.0 * x * v - 1.0), mu * (1.0 - x ** 2)],
                ]
            )

        sol = solve_ivp(
            vdp,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-5,
            atol=1e-8,
            jac=vdp_jac,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return the final state
        return sol.y[:, -1].tolist()