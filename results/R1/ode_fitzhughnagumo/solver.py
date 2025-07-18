import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def fitzhugh_nagumo(t, y, a, b, c, I):
    v, w = y
    dv_dt = v - (v**3)/3 - w + I
    dw_dt = a * (b * v - c * w)
    return np.array([dv_dt, dw_dt])

class Solver:
    def solve(self, problem, **kwargs):
        # Extract parameters
        t0, t1 = problem["t0"], problem["t1"]
        y0 = np.array(problem["y0"], dtype=np.float64)
        p = problem["params"]
        a = p["a"]
        b = p["b"]
        c = p["c"]
        I = p["I"]
        
        # Precompile the function
        fitzhugh_nagumo(np.array([0.0]), np.array([0.0, 0.0]), a, b, c, I)
        
        # Solve ODE
        sol = solve_ivp(
            fitzhugh_nagumo,
            [t0, t1],
            y0,
            args=(a, b, c, I),
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False
        )
        
        # Return the final state
        return sol.y[:, -1].tolist()