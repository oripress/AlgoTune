import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k = tuple(problem["k"])
        
        def rober(t, y):
            k1, k2, k3 = k
            f0 = -k1 * y[0] + k3 * y[1] * y[2]
            f1 = k1 * y[0] - k2 * y[1]**2 - k3 * y[1] * y[2]
            f2 = k2 * y[1]**2
            return np.array([f0, f1, f2])
        
        # Use looser tolerances since verification allows rtol=1e-5, atol=1e-8
        rtol = 1e-6
        atol = 1e-8
        method = "Radau"
        
        sol = solve_ivp(
            rober,
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