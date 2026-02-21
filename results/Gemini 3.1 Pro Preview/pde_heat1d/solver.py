import numpy as np
from scipy.integrate import RK45
from numba import njit

@njit(cache=True)
def _heat_equation_numba(u, u_padded, dx_sq, alpha):
    u_padded[1:-1] = u
    return alpha * ((u_padded[2:] - 2 * u + u_padded[:-2]) / dx_sq)

class Solver:
    def __init__(self):
        # Trigger Numba compilation
        _heat_equation_numba(np.zeros(10), np.zeros(12), 0.1, 0.1)

    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        
        n = len(y0)
        dx_sq = dx**2
        
        u_padded = np.zeros(n + 2)
        
        def heat_equation(t, u):
            return _heat_equation_numba(u, u_padded, dx_sq, alpha)
            
        solver = RK45(heat_equation, t0, y0, t1, rtol=1e-6, atol=1e-6)
        
        while solver.status == 'running':
            solver.step()
            
        if solver.status == 'finished':
            return solver.y.tolist()
        else:
            raise RuntimeError("Solver failed")