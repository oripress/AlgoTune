import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        N = params["num_points"]
        
        coeff = alpha / (dx * dx)
        
        def heat_equation(t, u):
            u_padded = np.empty(N + 2)
            u_padded[0] = 0.0
            u_padded[-1] = 0.0
            u_padded[1:-1] = u
            du_dt = coeff * (u_padded[2:] - 2.0 * u_padded[1:-1] + u_padded[:-2])
            return du_dt
        
        sol = solve_ivp(
            heat_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")