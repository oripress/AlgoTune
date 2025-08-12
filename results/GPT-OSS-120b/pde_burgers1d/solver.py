import numpy as np
from typing import Any, Dict, List
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the 1D Burgers' equation using SciPy's adaptive RK45 integrator.
        This matches the reference implementation in accuracy while being fast.
        """
        # Extract data
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        nu = float(params["nu"])
        dx = float(params["dx"])

        # Pre‑allocate padded array for Dirichlet boundary conditions (u=0)
        padded = np.empty(len(y0) + 2, dtype=float)

        def rhs(t, u_vec):
            """Compute du/dt for the current state."""
            # Apply Dirichlet boundaries by padding with zeros
            padded[0] = 0.0
            padded[-1] = 0.0
            padded[1:-1] = u_vec

            # Diffusion term (central second difference)
            diffusion = (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / (dx * dx)

            # Upwind advection term
            du_dx_forward = (padded[2:] - padded[1:-1]) / dx
            du_dx_backward = (padded[1:-1] - padded[:-2]) / dx

            advect = np.where(
                u_vec >= 0,
                u_vec * du_dx_backward,
                u_vec * du_dx_forward,
            )
            return -advect + nu * diffusion

        # Adaptive integration with tight tolerances
        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return the final state at t1
        return sol.y[:, -1].tolist()