from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit(fastmath=True)
def burgers_deriv(t, u, nu, dx):
    n = len(u)
    du_dt = np.empty(n, dtype=np.float64)
    
    dx2 = dx * dx
    inv_dx = 1.0 / dx
    inv_dx2 = 1.0 / dx2
    
    # i = 0
    u_val = u[0]
    u_left = 0.0
    u_right = u[1]
    
    diff = (u_right - 2.0 * u_val + u_left) * inv_dx2
    du_dx_fwd = (u_right - u_val) * inv_dx
    du_dx_bwd = (u_val - u_left) * inv_dx
    
    if u_val >= 0:
        adv = u_val * du_dx_bwd
    else:
        adv = u_val * du_dx_fwd
    du_dt[0] = -adv + nu * diff
    
    # Interior
    for i in range(1, n - 1):
        u_val = u[i]
        u_left = u[i-1]
        u_right = u[i+1]
        
        diff = (u_right - 2.0 * u_val + u_left) * inv_dx2
        du_dx_fwd = (u_right - u_val) * inv_dx
        du_dx_bwd = (u_val - u_left) * inv_dx
        
        if u_val >= 0:
            adv = u_val * du_dx_bwd
        else:
            adv = u_val * du_dx_fwd
        du_dt[i] = -adv + nu * diff
        
    # i = n - 1
    u_val = u[n-1]
    u_left = u[n-2]
    u_right = 0.0
    
    diff = (u_right - 2.0 * u_val + u_left) * inv_dx2
    du_dx_fwd = (u_right - u_val) * inv_dx
    du_dx_bwd = (u_val - u_left) * inv_dx
    
    if u_val >= 0:
        adv = u_val * du_dx_bwd
    else:
        adv = u_val * du_dx_fwd
    du_dt[n-1] = -adv + nu * diff
    
    return du_dt

class Solver:
    def __init__(self):
        # Warmup
        y0 = np.array([0.0, 0.1, 0.0], dtype=np.float64)
        burgers_deriv(0.0, y0, 0.01, 0.1)

    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu = float(params["nu"])
        dx = float(params["dx"])
        
        fun = lambda t, y: burgers_deriv(t, y, nu, dx)
        
        rtol = 1e-6
        atol = 1e-6
        method = "RK45"
        
        sol = solve_ivp(
            fun,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")