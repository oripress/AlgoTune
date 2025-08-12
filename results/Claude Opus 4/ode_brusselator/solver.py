import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

class Solver:
    def __init__(self):
        # Pre-compile the numba function
        self._brusselator_nb = self._make_brusselator_nb()
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _brusselator_core(y, A, B):
        X = y[0]
        Y = y[1]
        
        # Brusselator equations with optimized computation
        X2 = X * X
        X2Y = X2 * Y
        dX_dt = A + X2Y - (B + 1) * X
        dY_dt = B * X - X2Y
        
        return np.array([dX_dt, dY_dt])
    
    def _make_brusselator_nb(self):
        brusselator_core = self._brusselator_core
        
        def brusselator(t, y, A, B):
            return brusselator_core(y, A, B)
        
        return brusselator
    
    def solve(self, problem, **kwargs):
        """Solve the Brusselator model."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = problem["t0"]
        t1 = problem["t1"]
        A = problem["params"]["A"]
        B = problem["params"]["B"]
        
        # Use DOP853 - 8th order Runge-Kutta method
        # More efficient for smooth problems with tight tolerances
        sol = solve_ivp(
            self._brusselator_nb,
            [t0, t1],
            y0,
            args=(A, B),
            method='DOP853',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1]  # Return final state