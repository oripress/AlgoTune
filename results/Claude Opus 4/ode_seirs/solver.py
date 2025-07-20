import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the SEIRS epidemic model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # SEIRS model function
        def seirs(t, y):
            S, E, I, R = y
            beta = params["beta"]
            sigma = params["sigma"]
            gamma = params["gamma"]
            omega = params["omega"]
            
            # SEIRS equations
            dSdt = -beta * S * I + omega * R
            dEdt = beta * S * I - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I - omega * R
            
            return [dSdt, dEdt, dIdt, dRdt]
        
        # Use LSODA which is often faster for stiff problems
        # Relax tolerances slightly while maintaining accuracy
        sol = solve_ivp(
            seirs,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=5e-8,
            atol=5e-8,
            dense_output=False,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()