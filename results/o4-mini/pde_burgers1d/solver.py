from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        # Extract initial condition and parameters
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        params = problem["params"]
        nu = float(params["nu"])
        dx = float(params["dx"])

        # Define RHS of ODE system based on method of lines
        def burgers_equation(t, u):
            # Apply Dirichlet BC by padding
            u_padded = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)
            # Diffusion term (central difference)
            diffusion_term = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx * dx)
            # Advection term (upwind)
            u_center = u_padded[1:-1]
            du_dx_fwd = (u_padded[2:] - u_center) / dx
            du_dx_bwd = (u_center - u_padded[:-2]) / dx
            advection = np.where(u_center >= 0, u_center * du_dx_bwd, u_center * du_dx_fwd)
            # Combine terms
            return -advection + nu * diffusion_term

        # Solve with RK45 matching reference tolerances
        sol = solve_ivp(
            burgers_equation,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        # Return final state
        return sol.y[:, -1].tolist()