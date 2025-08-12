from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def __init__(self):
        """Initialize the solver."""
        pass
    
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Lorenz 96 system."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])
        N = len(y0)
        
        # Pre-compute index arrays for vectorized operations
        ip1 = np.roll(np.arange(N), -1)  # i+1 indices
        im1 = np.roll(np.arange(N), 1)   # i-1 indices  
        im2 = np.roll(np.arange(N), 2)   # i-2 indices
        
        def lorenz96(t, x):
            # Vectorized implementation
            dxdt = (x[ip1] - x[im2]) * x[im1] - x + F
            return dxdt
        
        # Use RK45 like the reference
        sol = solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-8,  # Use same tolerance as reference
            atol=1e-8,
            dense_output=False,
            t_eval=None
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")