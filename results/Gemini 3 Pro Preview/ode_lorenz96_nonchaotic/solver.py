import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def lorenz96_numba(t, x, F):
    N = len(x)
    dxdt = np.empty(N)
    # i=0
    dxdt[0] = (x[1] - x[N-2]) * x[N-1] - x[0] + F
    # i=1
    dxdt[1] = (x[2] - x[N-1]) * x[0] - x[1] + F
    # i=N-1
    dxdt[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1] + F
    
    for i in range(2, N-1):
        dxdt[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
    return dxdt

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])
        
        # Wrapper for solve_ivp
        # solve_ivp expects f(t, y)
        # We can pass args to it, but let's just use a lambda or partial if needed, 
        # but solve_ivp supports 'args' tuple.
        
        sol = solve_ivp(
            lorenz96_numba,
            (t0, t1),
            y0,
            args=(F,),
            method="RK45",
            rtol=1e-8,
            atol=1e-8
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")