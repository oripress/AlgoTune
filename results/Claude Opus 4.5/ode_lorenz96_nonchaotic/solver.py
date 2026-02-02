import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# Pre-compile the function outside the class
@njit(cache=True)
def lorenz96_derivative(x, ip1, im1, im2, F):
    N = len(x)
    dxdt = np.empty(N)
    for i in range(N):
        dxdt[i] = (x[ip1[i]] - x[im2[i]]) * x[im1[i]] - x[i] + F
    return dxdt

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])
        N = len(y0)
        
        # Precompute indices for cyclic boundary conditions
        ip1 = np.array([(i + 1) % N for i in range(N)], dtype=np.int64)
        im1 = np.array([(i - 1) % N for i in range(N)], dtype=np.int64)
        im2 = np.array([(i - 2) % N for i in range(N)], dtype=np.int64)
        
        # Warm up numba
        _ = lorenz96_derivative(y0, ip1, im1, im2, F)
        
        def lorenz96(t, x):
            return lorenz96_derivative(x, ip1, im1, im2, F)
        
        sol = solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")