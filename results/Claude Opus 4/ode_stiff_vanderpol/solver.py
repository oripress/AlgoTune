import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the stiff Van der Pol equation."""
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])
        
        def vdp(t, y):
            x, v = y
            dx_dt = v
            dv_dt = mu * ((1 - x**2) * v - x)
            return np.array([dx_dt, dv_dt])
        
        # Use BDF method which is often faster for stiff problems
        sol = solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method="BDF",
            rtol=1e-8,
            atol=1e-9,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")