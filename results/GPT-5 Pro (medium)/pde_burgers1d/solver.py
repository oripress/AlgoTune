from __future__ import annotations

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

        # Preallocate work arrays to minimize allocations in RHS
        u_left = np.empty_like(y0)
        u_right = np.empty_like(y0)
        du_dx_forward = np.empty_like(y0)
        du_dx_backward = np.empty_like(y0)
        diffusion_term = np.empty_like(y0)
        advection_term = np.empty_like(y0)
        du_dt = np.empty_like(y0)

        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx

        def rhs(_t: float, u: np.ndarray) -> np.ndarray:
            # Construct left/right with Dirichlet boundaries at 0
            # u_left = [0, u[0], u[1], ..., u[n-2]]
            u_left[0] = 0.0
            u_left[1:] = u[:-1]
            # u_right = [u[1], u[2], ..., u[n-1], 0]
            u_right[:-1] = u[1:]
            u_right[-1] = 0.0

            # Diffusion: central difference
            # (u_{i+1} - 2u_i + u_{i-1}) / dx^2
            np.subtract(u_right, u, out=diffusion_term)  # u_right - u
            np.subtract(diffusion_term, u, out=diffusion_term)  # u_right - 2u
            np.add(diffusion_term, u_left, out=diffusion_term)  # u_right - 2u + u_left
            np.multiply(diffusion_term, inv_dx2, out=diffusion_term)

            # Advection: upwind based on sign of u
            # forward/backward differences
            np.subtract(u_right, u, out=du_dx_forward)
            np.multiply(du_dx_forward, inv_dx, out=du_dx_forward)
            np.subtract(u, u_left, out=du_dx_backward)
            np.multiply(du_dx_backward, inv_dx, out=du_dx_backward)

            # Select upwind derivative into advection_term, then multiply by u
            mask = u < 0.0  # forward where u < 0, backward otherwise
            np.copyto(advection_term, du_dx_backward)
            np.copyto(advection_term, du_dx_forward, where=mask)
            np.multiply(advection_term, u, out=advection_term)

            # Combine: du/dt = -advection + nu * diffusion
            np.multiply(advection_term, -1.0, out=du_dt)
            np.multiply(diffusion_term, nu, out=diffusion_term)
            np.add(du_dt, diffusion_term, out=du_dt)

            # Return a copy to avoid any aliasing with solver internals
            return du_dt.copy()

        # Integrate using the same tolerances/method as the reference
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            t_eval=None,
            dense_output=False,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Final state
        return sol.y[:, -1].tolist()