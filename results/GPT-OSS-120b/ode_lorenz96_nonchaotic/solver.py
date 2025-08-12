import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Lorenz‑96 system using SciPy's adaptive RK45 integrator.
        Returns the state vector at the final time t1.
        """
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])

        def lorenz96(t, x):
            # (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
            return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

        # Use the same method and tolerances as the reference implementation,
        # adding a modest max_step to ensure sufficient resolution.
        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=[t1],
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Final state is the last column of sol.y
        final_state = sol.y[:, -1]
        return final_state.tolist()