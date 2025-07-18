from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the 1D Burgers' equation using the method of lines.

        The primary optimization is the Just-In-Time (JIT) compilation of the
        right-hand-side (RHS) function using Numba. The ODE solver from SciPy
        calls this RHS function many times. By compiling it to highly efficient
        machine code, we eliminate the Python interpreter overhead for the most
        computationally intensive part of the algorithm.

        The Numba-jitted function uses an explicit loop over the grid points.
        This pattern is very effectively optimized by Numba and avoids the
        creation of temporary intermediate arrays that a pure NumPy vectorized
        approach would use, further improving performance and memory usage.
        """
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]

        # This function is defined inside `solve` so that it can close over
        # the parameters `nu` and `dx` from the problem description.
        # Numba handles this closure efficiently.
        # `nopython=True` ensures we get maximum performance.
        # `fastmath=True` allows for aggressive floating-point optimizations.
        # `cache=True` caches the compiled code between runs.
        @numba.jit(nopython=True, fastmath=True, cache=True)
        def burgers_rhs_numba(t, u):
            N = u.shape[0]
            dudt = np.empty(N)
            dx2 = dx * dx

            # Handle the first grid point (i=0) with boundary condition u[-1]=0
            u_center = u[0]
            u_right = u[1]
            
            diffusion = nu * (u_right - 2 * u_center + 0.0) / dx2
            
            if u_center >= 0:
                # Upwind scheme: information comes from the left
                advection = u_center * (u_center - 0.0) / dx
            else:
                # Upwind scheme: information comes from the right
                advection = u_center * (u_right - u_center) / dx
            
            dudt[0] = diffusion - advection

            # Loop over interior grid points
            for i in range(1, N - 1):
                u_left = u[i-1]
                u_center = u[i]
                u_right = u[i+1]
                
                diffusion = nu * (u_right - 2 * u_center + u_left) / dx2
                
                if u_center >= 0:
                    advection = u_center * (u_center - u_left) / dx
                else:
                    advection = u_center * (u_right - u_center) / dx
                
                dudt[i] = diffusion - advection

            # Handle the last grid point (i=N-1) with boundary condition u[N]=0
            u_left = u[N-2]
            u_center = u[N-1]
            
            diffusion = nu * (0.0 - 2 * u_center + u_left) / dx2
            
            if u_center >= 0:
                advection = u_center * (u_center - u_left) / dx
            else:
                advection = u_center * (0.0 - u_center) / dx
                
            dudt[N-1] = diffusion - advection
            
            return dudt

        # Solve the ODE system using the JIT-compiled RHS function
        sol = solve_ivp(
            burgers_rhs_numba,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()