import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

@jit(nopython=True, cache=True)
def lorenz96_numba(t, x, F):
    """Numba-compiled Lorenz96 dynamics."""
    N = len(x)
    dxdt = np.empty(N, dtype=x.dtype)
    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        dxdt[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dxdt

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the Lorenz 96 system."""
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])
        
        def lorenz96(t, x):
            return lorenz96_numba(t, x, F)
        
        rtol = 1e-8
        atol = 1e-8
        method = "RK45"
        
        sol = solve_ivp(
            lorenz96,
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