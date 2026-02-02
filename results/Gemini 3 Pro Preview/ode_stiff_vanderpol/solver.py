import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        y0 = problem["y0"]
        t0 = problem["t0"]
        t1 = problem["t1"]
        mu = problem["mu"]
        
        # Define the system of ODEs
        def vdp(t, y):
            x, v = y
            return [v, mu * ((1 - x**2) * v - x)]
        
        # Define the Jacobian matrix
        def vdp_jac(t, y):
            x, v = y
            return [
                [0.0, 1.0],
                [mu * (-2.0 * x * v - 1.0), mu * (1.0 - x**2)]
            ]
        
        # Solve using Radau (same as reference but with Jacobian)
        sol = solve_ivp(
            vdp, 
            (t0, t1), 
            y0, 
            method='Radau', 
            jac=vdp_jac,
            rtol=1e-8, 
            atol=1e-9
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            # Fallback or raise error, though reference raises error
            raise RuntimeError("Solver failed")