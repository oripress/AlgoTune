import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
from numba import jit
import numba

# Precompile the Lotka-Volterra equations
@jit(nopython=True)
def lotka_volterra_numba(t, y, alpha, beta, delta, gamma):
    x, y_pop = y[0], y[1]
    dx_dt = x * (alpha - beta * y_pop)
    dy_dt = y_pop * (delta * x - gamma)
    return np.array([dx_dt, dy_dt])

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Solve the Lotka-Volterra predator-prey model efficiently."""
        # Extract parameters
        t0, t1 = problem["t0"], problem["t1"]
        y0 = np.array(problem["y0"], dtype=np.float64)
        params = problem["params"]
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        gamma = params["gamma"]
        
        # Define the Lotka-Volterra equations
        def lotka_volterra(t, y):
            return lotka_volterra_numba(t, y, alpha, beta, delta, gamma)
        
        # Solve with optimized parameters
        # Solve with optimized parameters
        sol = solve_ivp(
            lotka_volterra,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-9,
            atol=1e-9,
            t_eval=None,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")