import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

@njit(cache=True)
def burgers_rhs_numba(u, nu, inv_dx, inv_dx2, N):
    """Compute RHS of Burgers' equation using method of lines."""
    du_dt = np.empty(N)
    
    for i in range(N):
        u_left = u[i-1] if i > 0 else 0.0
        u_center = u[i]
        u_right = u[i+1] if i < N-1 else 0.0
        
        # Diffusion term (central difference)
        diffusion = (u_right - 2.0 * u_center + u_left) * inv_dx2
        
        # Advection term (upwind scheme)
        if u_center >= 0:
            du_dx = (u_center - u_left) * inv_dx
        else:
            du_dx = (u_right - u_center) * inv_dx
        
        du_dt[i] = -u_center * du_dx + nu * diffusion
    
    return du_dt

class Solver:
    def __init__(self):
        # Warm up numba JIT
        dummy = np.array([0.1, 0.2, 0.1])
        burgers_rhs_numba(dummy, 0.005, 10.0, 100.0, 3)
    
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        nu = float(params["nu"])
        dx = float(params["dx"])
        N = int(params["num_points"])
        
        inv_dx = 1.0 / dx
        inv_dx2 = 1.0 / (dx * dx)
        
        def rhs(t, u):
            return burgers_rhs_numba(u, nu, inv_dx, inv_dx2, N)
        
        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")