from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """Solve the 1D Burgers' equation."""
        # Extract problem parameters
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]
        
        # Define the Burgers' equation ODE system
        def burgers_equation(t, u):
            # Pre-allocate padded array to reduce memory allocation
            u_padded = np.empty(len(u) + 2)  # Two more for boundary
            u_padded[1:-1] = u
            u_padded[0] = 0  # Boundary conditions
            u_padded[-1] = 0  # Boundary conditions

            # Compute diffusion term (second derivative) using central difference
            # Reuse u_padded to avoid recomputation
            diffusion_term = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx**2)

            # Compute advection term using upwind scheme
            u_centered = u_padded[1:-1]  # u at current point

            # Forward difference (for u < 0)
            du_dx_forward = (u_padded[2:] - u_padded[1:-1]) / dx

            # Backward difference (for u > 0)
            du_dx_backward = (u_padded[1:-1] - u_padded[:-2]) / dx

            # Choose appropriate differencing based on sign of u (upwind scheme)
            advection_term = np.where(
                u_centered >= 0,
                u_centered * du_dx_backward,  # Use backward difference when u ≥ 0
                u_centered * du_dx_forward,    # Use forward difference when u < 0
            )

            # Combine terms: du/dt = -u*du/dx + nu*d²u/dx²
            du_dt = -advection_term + nu * diffusion_term

            return du_dt
        # Solve the ODE system
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            t_eval=None,
        )
        
        # Return the final state
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")