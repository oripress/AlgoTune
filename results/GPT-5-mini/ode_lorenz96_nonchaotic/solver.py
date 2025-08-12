from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the Lorenz-96 system:
            dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        with cyclic boundary conditions.

        Required problem keys: "F", "t0", "t1", "y0".
        Returns the state at time t1 as a list of floats.
        """
        if not {"F", "t0", "t1", "y0"}.issubset(problem):
            raise ValueError("Problem must contain keys 'F', 't0', 't1', 'y0'.")

        try:
            F = float(problem["F"])
            t0 = float(problem["t0"])
            t1 = float(problem["t1"])
            y0 = np.asarray(problem["y0"], dtype=np.float64).ravel()
        except Exception as e:
            raise ValueError(f"Invalid problem inputs: {e}")

        # Quick exits
        if y0.size == 0:
            return []
        if t0 == t1:
            return y0.tolist()

        N = y0.size
        idx = np.arange(N, dtype=int)
        ip1 = np.roll(idx, -1)
        im1 = np.roll(idx, 1)
        im2 = np.roll(idx, 2)

        def lorenz96(t, x):
            # Vectorized RHS evaluation
            return (x[ip1] - x[im2]) * x[im1] - x + F

        # Match reference solver tolerances/method by default for consistency
        method = kwargs.get("method", "RK45")
        rtol = float(kwargs.get("rtol", 1e-8))
        atol = float(kwargs.get("atol", 1e-8))

        # Integrate only to the final time to minimize overhead/memory
        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=[t1],
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return [float(v) for v in sol.y[:, -1]]