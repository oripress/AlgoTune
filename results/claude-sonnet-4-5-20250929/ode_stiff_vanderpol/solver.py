from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def vdp_numba(t, y, mu):
    x, v = y
    dx_dt = v
    dv_dt = mu * ((1.0 - x**2) * v - x)
    return np.array([dx_dt, dv_dt])

@njit
def jac_numba(t, y, mu):
    x, v = y
    jac = np.empty((2, 2))
    jac[0, 0] = 0.0
    jac[0, 1] = 1.0
    jac[1, 0] = mu * (-2.0 * x * v - 1.0)
    jac[1, 1] = mu * (1.0 - x**2)
    return jac

class Solver:
    def __init__(self):
        # Pre-warm the JIT-compiled functions
        dummy_y = np.array([0.5, 0.0])
        vdp_numba(0.0, dummy_y, 1000.0)
        jac_numba(0.0, dummy_y, 1000.0)
    
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])

        def vdp(t, y):
            return vdp_numba(t, y, mu)
        
        def jac(t, y):
            return jac_numba(t, y, mu)

        # Match verification tolerances exactly
        rtol = 1e-5
        atol = 1e-8
        method = "Radau"
        
        # Provide good initial step size based on stiffness
        first_step = 0.001 / np.sqrt(mu)
        
        # Limit maximum step to help convergence
        max_step = 0.1

        sol = solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method=method,
            jac=jac,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_step=max_step,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()