import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        
        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        omega = params["omega"]
        
        # Define derivative function inline for speed
        def seirs(t, y):
            S, E, I, R = y
            beta_S_I = beta * S * I
            omega_R = omega * R
            sigma_E = sigma * E
            gamma_I = gamma * I
            
            return np.array([
                -beta_S_I + omega_R,
                beta_S_I - sigma_E,
                sigma_E - gamma_I,
                gamma_I - omega_R
            ])
        
        # LSODA is often faster and auto-switches between stiff/non-stiff
        rtol = 1e-6
        atol = 1e-9
        
        sol = solve_ivp(
            seirs,
            [t0, t1],
            y0,
            method="LSODA",
            rtol=rtol,
            atol=atol,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")