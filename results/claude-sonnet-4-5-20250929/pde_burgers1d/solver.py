import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

@njit
def burgers_rhs(u, nu, dx_inv, dx2_inv):
    """Compute right-hand side of Burgers' equation with boundary conditions."""
    n = len(u)
    du_dt = np.empty(n)
    
    # First element (left boundary is 0)
    u_left = 0.0
    u_center = u[0]
    u_right = u[1] if n > 1 else 0.0
    
    diffusion = nu * (u_right - 2*u_center + u_left) * dx2_inv
    du_dx = (u_center - u_left) * dx_inv if u_center >= 0 else (u_right - u_center) * dx_inv
    du_dt[0] = -u_center * du_dx + diffusion
    
    # Middle elements
    for i in range(1, n-1):
        u_left = u[i-1]
        u_center = u[i]
        u_right = u[i+1]
        
        diffusion = nu * (u_right - 2*u_center + u_left) * dx2_inv
        du_dx = (u_center - u_left) * dx_inv if u_center >= 0 else (u_right - u_center) * dx_inv
        du_dt[i] = -u_center * du_dx + diffusion
    
    # Last element (right boundary is 0)
    if n > 1:
        u_left = u[n-2]
        u_center = u[n-1]
        u_right = 0.0
        
        diffusion = nu * (u_right - 2*u_center + u_left) * dx2_inv
        du_dx = (u_center - u_left) * dx_inv if u_center >= 0 else (u_right - u_center) * dx_inv
        du_dt[n-1] = -u_center * du_dx + diffusion
    
    return du_dt

class Solver:
    def solve(self, problem):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]
        dx_inv = 1.0 / dx
        dx2_inv = 1.0 / (dx * dx)
        
        def burgers_equation(t, u):
            return burgers_rhs(u, nu, dx_inv, dx2_inv)
        
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()