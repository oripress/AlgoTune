import numpy as np
from scipy.integrate import solve_ivp
import numba

@numba.njit
def burgers_equation_numba(t, u, nu, dx):
    n = len(u)
    du_dt = np.zeros(n)
    
    for i in range(n):
        # Diffusion term with boundary conditions
        if i == 0:
            diffusion = (u[1] - 2*u[0] + 0.0) / (dx**2)
        elif i == n-1:
            diffusion = (0.0 - 2*u[n-1] + u[n-2]) / (dx**2)
        else:
            diffusion = (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)
        
        # Advection term with upwind scheme
        if u[i] >= 0:
            if i == 0:
                advection = u[0] * (u[0] - 0.0) / dx
            else:
                advection = u[i] * (u[i] - u[i-1]) / dx
        else:
            if i == n-1:
                advection = u[i] * (0.0 - u[i]) / dx
            else:
                advection = u[i] * (u[i+1] - u[i]) / dx
        
        du_dt[i] = -advection + nu * diffusion
    
    return du_dt

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]
        
        def burgers_equation(t, u):
            return burgers_equation_numba(t, u, nu, dx)
        
        rtol = 1e-6
        atol = 1e-6
        method = "RK45"
        
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()