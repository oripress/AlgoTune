import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the Lotka-Volterra predator-prey model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        def lotka_volterra(t, y):
            x, y_pred = y
            alpha = params["alpha"]
            beta = params["beta"]
            delta = params["delta"]
            gamma = params["gamma"]
            
            dx_dt = alpha * x - beta * x * y_pred
            dy_dt = delta * x * y_pred - gamma * y_pred
            
            return np.array([dx_dt, dy_dt])
        
        # Use RK45 with reasonable tolerances
        sol = solve_ivp(
            lotka_volterra,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")