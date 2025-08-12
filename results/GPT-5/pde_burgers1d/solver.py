from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        # Extract inputs
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        nu = float(params["nu"])
        dx = float(params["dx"])

        n = y0.shape[0]
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx

        # Preallocate working arrays to minimize allocations per RHS call
        left = np.empty(n, dtype=float)
        right = np.empty(n, dtype=float)
        fwd = np.empty(n, dtype=float)
        bwd = np.empty(n, dtype=float)
        adv = np.empty(n, dtype=float)
        du_dt = np.empty(n, dtype=float)
        mask = np.empty(n, dtype=bool)

        def rhs(_t, u):
            # Apply Dirichlet boundary conditions via shifted neighbors
            # left neighbor (with 0 at the left boundary)
            left[0] = 0.0
            left[1:] = u[:-1]
            # right neighbor (with 0 at the right boundary)
            right[-1] = 0.0
            right[:-1] = u[1:]

            # Diffusion term: (u_{i+1} - 2u_i + u_{i-1}) / dx^2
            # du_dt temporarily stores second derivative
            # du_dt = right - u
            np.subtract(right, u, out=du_dt)
            # du_dt = (right - u) - u
            np.subtract(du_dt, u, out=du_dt)
            # du_dt = (right - 2u) + left
            np.add(du_dt, left, out=du_dt)
            # scale by 1/dx^2
            np.multiply(du_dt, inv_dx2, out=du_dt)

            # Advection term with upwind scheme
            # forward difference (for u < 0): (right - u) / dx
            np.subtract(right, u, out=fwd)
            np.multiply(fwd, inv_dx, out=fwd)
            # backward difference (for u >= 0): (u - left) / dx
            np.subtract(u, left, out=bwd)
            np.multiply(bwd, inv_dx, out=bwd)

            # Select upwind gradient based on sign of u
            # adv = u * (u>=0 ? bwd : fwd)
            np.copyto(adv, bwd)
            np.signbit(u, out=mask)  # True where u < 0
            adv[mask] = fwd[mask]
            np.multiply(adv, u, out=adv)

            # Combine: du/dt = -advection + nu * diffusion
            np.multiply(du_dt, nu, out=du_dt)
            np.subtract(du_dt, adv, out=du_dt)
            return du_dt.copy()

        # RK45 for compatibility, only request final point to reduce overhead
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            t_eval=[t1],
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Final state at t1
        return sol.y[:, -1].tolist()