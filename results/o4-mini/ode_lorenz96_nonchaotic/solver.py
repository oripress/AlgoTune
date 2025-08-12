import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract input parameters
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem.get("t0", 0.0))
        t1 = float(problem.get("t1", 0.0))
        F = float(problem.get("F", 0.0))
        # Number of variables
        N = y0.shape[0]
        # Precompute cyclic index arrays
        ip1 = np.arange(N, dtype=int) + 1
        ip1[-1] = 0
        im1 = np.arange(N, dtype=int) - 1
        im1[0] = N - 1
        im2 = np.arange(N, dtype=int) - 2
        im2[0] = N - 2
        im2[1] = N - 1
        # Define Lorenz-96 ODE
        def lorenz96(t, x):
            return (x[ip1] - x[im2]) * x[im1] - x + F
        # Integrate ODE
        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            vectorized=False,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        # Return final state as list
        return sol.y[:, -1].tolist()