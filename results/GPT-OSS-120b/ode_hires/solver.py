import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, List, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the HIRES stiff ODE system from t0 to t1.
        Returns the state vector at the final time.
        """
        # Extract data
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        constants = np.asarray(problem["constants"], dtype=float)

        # Unpack constants for speed
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants

        # Simple RHS without JIT; SciPy's solver overhead dominates.
        # We'll accelerate the RHS using Numba JIT compilation.
        from numba import njit

        @njit(cache=True)
        def _hires_rhs(y, c):
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = c

            f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            f2 = c1 * y1 - c5 * y2
            f3 = -c6 * y3 + c2 * y4 + c7 * y5
            f4 = c3 * y2 + c1 * y3 - c8 * y4
            f5 = -c9 * y5 + c2 * y6 + c2 * y7
            f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            f7 = c10 * y6 * y8 - c12 * y7
            f8 = -c10 * y6 * y8 + c12 * y7

            return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

        def hires(t, y):
            # Wrapper that supplies the constant vector to the JIT‑compiled RHS
            return _hires_rhs(y, constants)
        # Use a stiff solver with tighter tolerances matching the reference implementation.
        # No max_step restriction to let the integrator adapt appropriately.
        sol = solve_ivp(
            hires,
            [t0, t1],
            y0,
            method="Radau",
            rtol=1e-10,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # sol.y has shape (n, len(t)), final column is the state at t1
        final_state = sol.y[:, -1]
        return final_state.tolist()