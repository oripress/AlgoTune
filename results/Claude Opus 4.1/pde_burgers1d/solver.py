import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
import numba as nb

@nb.njit(cache=True, fastmath=True)
def compute_burgers_rhs(u, nu, dx):
    """Numba-compiled right-hand side computation."""
    n = len(u)
    du_dt = np.zeros_like(u)
    
    # Pre-compute constants
    dx_inv = 1.0 / dx
    dx_sq_inv = 1.0 / (dx * dx)
    
    # Boundary conditions (u=0)
    # First point
    if n > 0:
        u_left = 0.0
        u_curr = u[0]
        u_right = u[1] if n > 1 else 0.0
        
        # Diffusion
        diffusion = nu * (u_right - 2*u_curr + u_left) * dx_sq_inv
        
        # Upwind advection
        if u_curr >= 0:
            advection = u_curr * (u_curr - u_left) * dx_inv
        else:
            advection = u_curr * (u_right - u_curr) * dx_inv
        
        du_dt[0] = -advection + diffusion
    
    # Interior points
    for i in range(1, n-1):
        u_left = u[i-1]
        u_curr = u[i]
        u_right = u[i+1]
        
        # Diffusion
        diffusion = nu * (u_right - 2*u_curr + u_left) * dx_sq_inv
        
        # Upwind advection
        if u_curr >= 0:
            advection = u_curr * (u_curr - u_left) * dx_inv
        else:
            advection = u_curr * (u_right - u_curr) * dx_inv
        
        du_dt[i] = -advection + diffusion
    
    # Last point
    if n > 1:
        u_left = u[n-2]
        u_curr = u[n-1]
        u_right = 0.0  # boundary
        
        # Diffusion
        diffusion = nu * (u_right - 2*u_curr + u_left) * dx_sq_inv
        
        # Upwind advection
        if u_curr >= 0:
            advection = u_curr * (u_curr - u_left) * dx_inv
        else:
            advection = u_curr * (u_right - u_curr) * dx_inv
        
        du_dt[n-1] = -advection + diffusion
    
    return du_dt

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the 1D Burgers' equation using optimized methods."""
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        nu = np.float64(problem["params"]["nu"])
        dx = np.float64(problem["params"]["dx"])
        
        # Force JIT compilation
        _ = compute_burgers_rhs(y0, nu, dx)
        
        def burgers_equation(t, u):
            return compute_burgers_rhs(u, nu, dx)
        
        # Use RK45 as in reference
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            dense_output=False,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")