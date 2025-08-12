import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
from typing import Any

@jit(nopython=True)
def heat_equation_numba(t, u, coeff):
    # Apply method of lines: discretize spatial derivatives using finite differences
    # For interior points, we use the standard central difference formula
    # Use padding to handle boundary conditions (u=0 at boundaries)
    n = len(u)
    u_padded = np.zeros(n + 2)
    u_padded[1:-1] = u  # Copy u into the padded array (boundaries are already 0)
    
    # Compute second derivative using central difference
    u_xx = np.empty(n)
    for i in range(n):
        u_xx[i] = (u_padded[i+2] - 2 * u_padded[i+1] + u_padded[i]) * coeff
    
    return u_xx

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)  # Ensure float64 for accuracy
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # Precompute constants
        alpha = params["alpha"]
        dx = params["dx"]
        dx2_inv = 1.0 / (dx * dx)  # Precompute inverse for efficiency
        
        # Precompute coefficient
        coeff = alpha * dx2_inv
        
        def heat_equation(t, u):
            return heat_equation_numba(t, u, coeff)
        
        # Set solver parameters to match reference exactly
        rtol = 1e-6
        atol = 1e-6
        method = "RK45"
        
        sol = solve_ivp(
            heat_equation,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")