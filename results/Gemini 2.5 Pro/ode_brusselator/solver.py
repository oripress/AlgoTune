from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

# CRITICAL CHANGE: The ODE function is decorated with @numba.jit.
# This compiles the function to highly optimized machine code, eliminating
# the Python function call overhead, which is the main bottleneck when using
# SciPy's solvers. `nopython=True` ensures the entire function is compiled
# without falling back to slower Python objects.
@jit(nopython=True)
def brusselator_ode(t, y, A, B):
    # The function signature is changed to accept A and B directly,
    # as Numba works best with simple scalar arguments, and solve_ivp
    # will unpack the `args` tuple automatically.
    X, Y = y
    dX_dt = A + X**2 * Y - (B + 1) * X
    dY_dt = B * X - X**2 * Y
    return np.array([dX_dt, dY_dt])

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        A highly optimized solver for the Brusselator ODE system using Numba and SciPy.

        This implementation leverages:
        1. Numba: A high-performance JIT compiler for numerical Python code.
           The ODE function itself is compiled, removing the primary performance
           bottleneck of Python function call overhead from within the solver's loop.
        2. SciPy's `solve_ivp` with `RK45`: This uses the exact same robust,
           battle-tested numerical algorithm as the reference solution, ensuring
           that the results are numerically identical while being much faster.
        """
        y0 = np.array(problem["y0"], dtype=np.float64)
        t_span = (float(problem["t0"]), float(problem["t1"]))
        params = problem["params"]
        args = (params["A"], params["B"])
        
        rtol = 1e-8
        atol = 1e-8

        # Call solve_ivp, passing the Numba-jitted ODE function.
        # The first run will incur a small compilation cost, but subsequent
        # runs will be significantly faster.
        sol = solve_ivp(
            fun=brusselator_ode,
            t_span=t_span,
            y0=y0,
            method='RK45',
            args=args,
            rtol=rtol,
            atol=atol,
        )
        
        # The result `sol.y` contains the solution at all steps. We need the
        # last column, which corresponds to the solution at t_span[1].
        return sol.y[:, -1]