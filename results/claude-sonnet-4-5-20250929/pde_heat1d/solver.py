import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def compute_derivative(u, alpha_over_dx2):
    n = len(u)
    du_dt = np.zeros(n)
    
    # First point (boundary at left is 0)
    du_dt[0] = alpha_over_dx2 * (u[1] - 2*u[0])
    
    # Interior points
    for i in range(1, n-1):
        du_dt[i] = alpha_over_dx2 * (u[i+1] - 2*u[i] + u[i-1])
    
    # Last point (boundary at right is 0)
    du_dt[n-1] = alpha_over_dx2 * (u[n-2] - 2*u[n-1])
    
    return du_dt

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        
        # Precompute coefficient
        alpha_over_dx2 = alpha / (dx * dx)
        
        def heat_equation(t, u):
            return compute_derivative(u, alpha_over_dx2)
        
        # Set solver parameters
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
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()