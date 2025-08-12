import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the 1D heat equation using method of lines."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        def heat_equation(t, u):
            # Extract parameters
            alpha = params["alpha"]
            dx = params["dx"]
            
            # Apply method of lines with boundary conditions (u=0 at boundaries)
            u_padded = np.pad(u, 1, mode="constant", constant_values=0)
            
            # Compute second derivative using central difference
            u_xx = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx**2)
            
            # Apply diffusion equation
            du_dt = alpha * u_xx
            
            return du_dt
        
        # Set solver parameters
        rtol = 1e-6
        atol = 1e-6
        
        sol = solve_ivp(
            heat_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")