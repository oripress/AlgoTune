from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        # Extract and validate inputs
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])

        n = y0.shape[0]
        idx = np.arange(n)
        ip1 = (idx + 1) % n
        im1 = (idx - 1) % n
        im2 = (idx - 2) % n

        # Vectorized Lorenz-96 RHS. Handles both (n,) and (n, m) shaped x.
        def lorenz96(t: float, x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                # x: (n,)
                return (x[ip1] - x[im2]) * x[im1] - x + F
            # x: (n, m)
            return (x[ip1, :] - x[im2, :]) * x[im1, :] - x + F

        # Match reference tolerances and method, but enable vectorized evaluations
        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            vectorized=True,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()