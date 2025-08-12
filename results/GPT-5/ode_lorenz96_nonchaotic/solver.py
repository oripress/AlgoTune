from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solve the Lorenz-96 ODE with cyclic boundary conditions using SciPy's RK45,
        matching the reference solver's configuration while minimizing per-call
        allocations in the RHS via a slicing-based shift and a single scratch buffer.
        """
        F = float(problem["F"])
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.asarray(problem["y0"], dtype=float)

        if t1 == t0:
            return y0.tolist()

        N = y0.size
        F_val = F  # local alias for speed

        # Preallocate scratch buffer for shifted vectors
        scratch = np.empty(N, dtype=y0.dtype)

        def lorenz96(_t: float, x: np.ndarray) -> np.ndarray:
            # Compute dxdt = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F using fast slicing shifts
            dxdt = np.empty_like(x)

            # dxdt <- x shifted by -1 (i+1)
            dxdt[:-1] = x[1:]
            dxdt[-1] = x[0]

            # scratch <- x shifted by +2 (i-2)
            if N > 2:
                scratch[2:] = x[:-2]
                scratch[:2] = x[-2:]
            else:
                # For N <= 2, shift by 2 is identity
                scratch[:] = x

            # dxdt <- dxdt - scratch
            np.subtract(dxdt, scratch, out=dxdt)

            # scratch <- x shifted by +1 (i-1)
            scratch[1:] = x[:-1]
            scratch[0] = x[-1]

            # dxdt <- dxdt * scratch
            np.multiply(dxdt, scratch, out=dxdt)

            # dxdt <- dxdt - x + F
            np.subtract(dxdt, x, out=dxdt)
            dxdt += F_val
            return dxdt

        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()