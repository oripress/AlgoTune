import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract initial data
        y0 = np.array(problem["y0"], dtype=float)
        t0 = problem["t0"]
        t1 = problem["t1"]
        # If no time evolution needed
        if t1 == t0:
            return y0.tolist()
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        inv_dx2 = 1.0 / (dx * dx)
        # Define ODE function
        def heat_equation(t, u):
            du = np.empty_like(u)
            # Left boundary (u=0)
            du[0] = alpha * (u[1] - 2 * u[0]) * inv_dx2
            # Right boundary (u=0)
            du[-1] = alpha * (-2 * u[-1] + u[-2]) * inv_dx2
            # Interior points
            if u.shape[0] > 2:
                du[1:-1] = alpha * (u[2:] - 2 * u[1:-1] + u[:-2]) * inv_dx2
            return du
        # Solve ODE
        sol = solve_ivp(
            heat_equation,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        # Return final state
        return sol.y[:, -1].tolist()