from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
import numba

# By defining the function at the top level and applying the @numba.njit
# decorator, Numba compiles it to fast machine code. This compiled
# function is then passed to the SciPy solver.
@numba.njit
def lorenz96(t, x, F, ip1, im1, im2):
    """
    Lorenz 96 dynamics, compiled for speed with Numba.
    This function is called repeatedly by the ODE solver.
    """
    # The core vectorized computation is identical to the original,
    # but Numba compiles this NumPy-like code into a highly efficient loop,
    # removing Python interpreter overhead.
    return (x[ip1] - x[im2]) * x[im1] - x + F

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the Lorenz 96 system of ODEs using a Numba-jitted RHS.
        """
        y0 = np.array(problem["y0"])
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])
        N = len(y0)

        # Pre-compute indices for the vectorized implementation.
        # These are passed as arguments to the jitted function.
        ip1 = np.roll(np.arange(N), -1)
        im1 = np.roll(np.arange(N), 1)
        im2 = np.roll(np.arange(N), 2)

        # solve_ivp calls the fast, compiled lorenz96 function.
        # The constant parameters (F and indices) are passed efficiently
        # via the `args` tuple, avoiding closure overhead.
        sol = solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method="RK45",
            args=(F, ip1, im1, im2),
            rtol=1e-8,
            atol=1e-8,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return the final state as a list.
        return sol.y[:, -1].tolist()