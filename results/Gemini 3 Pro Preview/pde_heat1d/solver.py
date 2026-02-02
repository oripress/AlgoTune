import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def heat_equation_numba(t, u, alpha, dx_sq_inv):
    n = len(u)
    du_dt = np.empty(n, dtype=np.float64)
    
    if n == 0:
        return du_dt
        
    if n == 1:
        du_dt[0] = alpha * (-2 * u[0]) * dx_sq_inv
        return du_dt

    # Left boundary (u[-1] = 0)
    du_dt[0] = alpha * (u[1] - 2 * u[0]) * dx_sq_inv
    
    # Interior
    for i in range(1, n - 1):
        du_dt[i] = alpha * (u[i+1] - 2 * u[i] + u[i-1]) * dx_sq_inv
        
    # Right boundary (u[n] = 0)
    du_dt[n-1] = alpha * (-2 * u[n-1] + u[n-2]) * dx_sq_inv
    
    return du_dt

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        
        dx_sq_inv = 1.0 / (dx**2)
        
        sol = solve_ivp(
            heat_equation_numba,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            args=(alpha, dx_sq_inv)
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")