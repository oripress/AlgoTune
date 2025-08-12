from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Solve the HIRES kinetics ODE system efficiently."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        constants = problem["constants"]
        
        # Define the HIRES ODE system function
        def hires(t, y):
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            
            # HIRES system of equations - vectorized computation
            dy = np.empty_like(y)
            dy[0] = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            dy[1] = c1 * y1 - c5 * y2
            dy[2] = -c6 * y3 + c2 * y4 + c7 * y5
            dy[3] = c3 * y2 + c1 * y3 - c8 * y4
            dy[4] = -c9 * y5 + c2 * y6 + c2 * y7
            dy[5] = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            dy[6] = c10 * y6 * y8 - c12 * y7
            dy[7] = -c10 * y6 * y8 + c12 * y7
            
            return dy
        
        # Use a more efficient approach
        sol = solve_ivp(
            hires,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10,
            first_step=1e-10
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")