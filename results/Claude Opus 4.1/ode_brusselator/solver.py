import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def __init__(self):
        # Pre-compile the derivative function
        @numba.njit
        def brusselator_jit(t, y, A, B):
            X = y[0]
            Y = y[1]
            
            X2 = X * X
            X2Y = X2 * Y
            
            dX_dt = A + X2Y - (B + 1) * X
            dY_dt = B * X - X2Y
            
            return np.array([dX_dt, dY_dt])
        
        self.brusselator_jit = brusselator_jit
    
    def solve(self, problem, **kwargs):
        """Solve the Brusselator model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        A = problem["params"]["A"]
        B = problem["params"]["B"]
        
        # Wrapper for scipy
        def brusselator(t, y):
            return self.brusselator_jit(t, y, A, B)
        
        # Adaptive parameters based on time span
        time_span = t1 - t0
        
        # RK45 is generally faster for non-stiff problems
        # Increase step sizes for longer simulations
        if time_span > 100:
            max_step = time_span / 50
        else:
            max_step = np.inf
            
        sol = solve_ivp(
            brusselator,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            max_step=max_step,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1]  # Return final state