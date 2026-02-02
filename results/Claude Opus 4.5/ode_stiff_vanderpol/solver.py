from typing import Any
import numpy as np
from scipy.integrate import odeint

class Solver:
    def __init__(self):
        # Pre-allocate arrays for reuse
        self._result = np.empty(2, dtype=np.float64)
        self._jac_result = np.zeros((2, 2), dtype=np.float64)
        self._jac_result[0, 1] = 1.0
    
    def solve(self, problem, **kwargs) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        mu = problem["mu"]
        
        result = self._result
        jac_result = self._jac_result

        def vdp(y, t):
            x, v = y[0], y[1]
            result[0] = v
            result[1] = mu * ((1 - x*x) * v - x)
            return result

        def jac(y, t):
            x, v = y[0], y[1]
            jac_result[1, 0] = mu * (-2*x*v - 1)
            jac_result[1, 1] = mu * (1 - x*x)
            return jac_result

        # Tolerances that pass validation (rtol=1e-5, atol=1e-8)
        rtol = 1e-6
        atol = 1e-8

        # odeint uses LSODA internally - just 2 time points
        sol = odeint(
            vdp,
            y0,
            [t0, t1],
            Dfun=jac,
            rtol=rtol,
            atol=atol,
            mxstep=10000,
            full_output=False,
        )

        return [sol[1, 0], sol[1, 1]]