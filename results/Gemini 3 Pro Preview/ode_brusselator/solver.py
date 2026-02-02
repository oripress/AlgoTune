from typing import Any
import numpy as np
from scipy.integrate import ode
from numba import njit

@njit(fastmath=True)
def brusselator(t, y, A, B):
    X = y[0]
    Y = y[1]
    X2 = X * X
    dX_dt = A + X2 * Y - (B + 1) * X
    dY_dt = B * X - X2 * Y
    return np.array([dX_dt, dY_dt])

class Solver:
    def __init__(self):
        # Trigger compilation
        brusselator(0.0, np.array([1.0, 1.0]), 1.0, 3.0)

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        A = params["A"]
        B = params["B"]

        # dopri5 is equivalent to RK45 (Dormand-Prince 4(5))
        r = ode(lambda t, y, A, B: brusselator(t, y, A, B)).set_integrator("dop853", rtol=1e-8, atol=1e-8, nsteps=1000000)
        r.set_initial_value(y0, t0)
        r.set_f_params(A, B)
        
        # Integrate to t1
        res = r.integrate(t1)
        
        if r.successful():
            return res
        else:
            raise RuntimeError("Solver failed")