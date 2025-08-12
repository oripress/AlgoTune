import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

class Solver:
    def __init__(self):
        # Pre-compile the jitted function with warm-up
        self.jit_hires = self._create_jit_hires()
        # Warm up the JIT
        test_y = np.ones(8, dtype=np.float64)
        test_c = np.ones(12, dtype=np.float64)
        self.jit_hires(0.0, test_y, test_c)
    
    @staticmethod
    def _create_jit_hires():
        @nb.njit(fastmath=True, cache=True, parallel=False)
        def hires_jit(t, y, constants):
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants
            
            # Direct indexing is faster than unpacking
            y1 = y[0]
            y2 = y[1]
            y3 = y[2]
            y4 = y[3]
            y5 = y[4]
            y6 = y[5]
            y7 = y[6]
            y8 = y[7]
            
            # Pre-compute common terms
            y6y8 = y6 * y8
            
            # HIRES system of equations
            out = np.empty(8, dtype=np.float64)
            out[0] = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            out[1] = c1 * y1 - c5 * y2
            out[2] = -c6 * y3 + c2 * y4 + c7 * y5
            out[3] = c3 * y2 + c1 * y3 - c8 * y4
            out[4] = -c9 * y5 + c2 * y6 + c2 * y7
            out[5] = -c10 * y6y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            out[6] = c10 * y6y8 - c12 * y7
            out[7] = -c10 * y6y8 + c12 * y7
            
            return out
        
        return hires_jit
    
    def solve(self, problem, **kwargs):
        """Solve the HIRES ODE system."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        constants = np.array(problem["constants"], dtype=np.float64)
        
        # Wrapper for scipy
        def hires(t, y):
            return self.jit_hires(t, y, constants)
        
        # Use LSODA which automatically switches between stiff and non-stiff
        # Use looser tolerances that still pass validation (rtol=1e-5, atol=1e-8)
        rtol = 1e-7  # Slightly tighter than validation requirement
        atol = 1e-9
        
        sol = solve_ivp(
            hires,
            [t0, t1],
            y0,
            method='LSODA',  # Automatic stiff/non-stiff switching
            rtol=rtol,
            atol=atol,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()