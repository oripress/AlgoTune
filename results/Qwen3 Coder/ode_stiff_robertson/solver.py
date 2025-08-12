from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Robertson chemical kinetics system."""
        # Extract parameters
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k1, k2, k3 = problem["k"]
        
        # Define the ODE system
        def rober(t, y):
            y1, y2, y3 = y
            f0 = -k1 * y1 + k3 * y2 * y3
            f1 = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
            f2 = k2 * y2**2
            return np.array([f0, f1, f2])
        
        # Solve the ODE with LSODA method and optimized parameters
        sol = solve_ivp(
            rober,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-6,
            atol=1e-9,
            t_eval=[t1],  # Only evaluate at final time
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")