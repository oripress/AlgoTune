import numpy as np
from scipy.integrate._ivp.rk import RK45
import numba as nb

@nb.njit(fastmath=True)
def fitzhugh_nagumo(t, y, a, b, c, I):
    v, w = y[0], y[1]
    dv_dt = v - (v**3) / 3.0 - w + I
    dw_dt = a * (b * v - c * w)
    return np.array([dv_dt, dw_dt])

class Solver:
    def __init__(self):
        # Pre-compile the Numba function
        fitzhugh_nagumo(0.0, np.array([0.0, 0.0]), 0.1, 0.1, 0.1, 0.1)

    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])

        def fun(t, y):
            return fitzhugh_nagumo(t, y, a, b, c, I)

        solver = RK45(fun, t0, y0, t1, rtol=1e-8, atol=1e-8)
        
        while solver.status == 'running':
            solver.step()
            
        return solver.y.tolist()