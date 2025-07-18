import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Parse inputs
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]
        dx2 = dx * dx

        # Define RHS for Burgers' equation
        def rhs(t, u):
            # Apply Dirichlet boundary conditions via padding
            up = np.empty(u.size + 2, dtype=np.float64)
            up[0] = 0.0
            up[-1] = 0.0
            up[1:-1] = u
            # Diffusion term: second derivative
            diff = (up[2:] - 2.0 * up[1:-1] + up[:-2]) / dx2
            # Advection term: upwind differencing
            uc = up[1:-1]
            dxf = (up[2:] - uc) / dx
            dxb = (uc - up[:-2]) / dx
            adv = np.where(uc >= 0.0, uc * dxb, uc * dxf)
            return -adv + nu * diff

        # Integrate using RK45
        sol = solve_ivp(
            rhs,
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