import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit(cache=True)
def heat_equation_rhs(u, alpha_dx2):
    n = len(u)
    du_dt = np.empty(n)
    
    # Handle boundary conditions (u=0 at boundaries)
    du_dt[0] = alpha_dx2 * (u[1] - 2*u[0])
    for i in range(1, n-1):
        du_dt[i] = alpha_dx2 * (u[i+1] - 2*u[i] + u[i-1])
    du_dt[n-1] = alpha_dx2 * (-2*u[n-1] + u[n-2])
    
    return du_dt

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        params = problem["params"]
        
        alpha = float(params["alpha"])
        dx = float(params["dx"])
        alpha_dx2 = alpha / (dx * dx)
        
        def heat_equation(t, u):
            return heat_equation_rhs(u, alpha_dx2)
        
        sol = solve_ivp(
            heat_equation,
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