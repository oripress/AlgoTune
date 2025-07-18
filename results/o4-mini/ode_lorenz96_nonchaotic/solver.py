import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract problem parameters
        F = float(problem["F"])
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.array(problem["y0"], dtype=np.float64)

        # Precompute indices for cyclic boundary conditions
        N = y0.size
        ip1 = np.concatenate((np.arange(1, N), [0]))
        im1 = np.concatenate(([N - 1], np.arange(0, N - 1)))
        im2 = np.concatenate(([N - 2, N - 1], np.arange(0, N - 2)))

        # Define the Lorenz-96 RHS using precomputed indices
        def lorenz96(t, x):
            return (x[ip1] - x[im2]) * x[im1] - x + F

        # Solve ODE with a high-order solver for efficiency and accuracy
        sol = solve_ivp(
            lorenz96, (t0, t1), y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        # Return the final state
        return sol.y[:, -1].tolist()