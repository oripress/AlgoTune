import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])
        N = len(y0)
        
        # Optimized Numba-accelerated ODE function
        @nb.njit(nb.float64[:](nb.float64, nb.float64[:]), fastmath=True, cache=True)
        def lorenz96(t, x):
            dxdt = np.empty_like(x)
            for i in range(N):
                # Calculate indices with modulo for cyclic boundaries
                im1 = (i - 1) % N
                im2 = (i - 2) % N
                ip1 = (i + 1) % N
                dxdt[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
            return dxdt
        
        # Use RK45 solver to ensure numerical consistency
        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            t_eval=[t1]  # Only compute solution at final time
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()