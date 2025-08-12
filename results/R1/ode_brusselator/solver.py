import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

@jit(nopython=True, fastmath=True, cache=True)
def brusselator(t, y, A, B):
    X, Y = y
    dX_dt = A + X**2 * Y - (B + 1) * X
    dY_dt = B * X - X**2 * Y
    return np.array([dX_dt, dY_dt])

class Solver:
    def solve(self, problem, **kwargs):
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"])
        A = problem["params"]["A"]
        B = problem["params"]["B"]
        
        # Precompile the function with initial values
        brusselator(t0, y0, A, B)
        
        # Calculate optimal step sizes based on time span
        time_span = t1 - t0
        max_step = max(0.01, min(0.5, time_span / 50))
        initial_step = max(0.001, min(0.1, time_span / 500))
        
        # Use optimized solver with strict tolerance
        sol = solve_ivp(
            brusselator,
            (t0, t1),
            y0,
            args=(A, B),
            method='DOP853',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            first_step=initial_step,
            max_step=max_step
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback with default step sizes if needed
            sol = solve_ivp(
                brusselator,
                (t0, t1),
                y0,
                args=(A, B),
                method='DOP853',
                rtol=1e-8,
                atol=1e-8
            )
            return sol.y[:, -1]