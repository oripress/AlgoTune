import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# Pre-compile the ODE function outside the class
@njit(cache=True)
def fitzhugh_nagumo_jit(y, a, b, c, I):
    v = y[0]
    w = y[1]
    result = np.empty(2)
    result[0] = v - (v*v*v) / 3.0 - w + I
    result[1] = a * (b * v - c * w)
    return result

class Solver:
    def __init__(self):
        # Warm up the JIT compilation
        dummy_y = np.array([0.0, 0.0])
        _ = fitzhugh_nagumo_jit(dummy_y, 0.1, 0.1, 0.1, 0.1)
    
    def solve(self, problem, **kwargs):
        """Solve the FitzHugh-Nagumo differential equation system."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # Extract parameters
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])
        
        # Wrapper for scipy that uses pre-compiled function
        def fitzhugh_nagumo(t, y):
            return fitzhugh_nagumo_jit(y, a, b, c, I)
        
        # Solve the ODE with RK45 for accuracy
        sol = solve_ivp(
            fitzhugh_nagumo,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            first_step=0.01,  # Hint for initial step size
            max_step=10.0     # Allow larger steps when possible
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        # Return final state as list
        return sol.y[:, -1].tolist()