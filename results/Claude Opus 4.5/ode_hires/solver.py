from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
import numba

# Module-level JIT compiled functions - compiled once
@numba.njit(cache=True, fastmath=True)
def hires_numba(t, y, c):
    y1, y2, y3, y4, y5, y6, y7, y8 = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]
    
    result = np.empty(8)
    result[0] = -c1 * y1 + c2 * y2 + c3 * y3 + c4
    result[1] = c1 * y1 - c5 * y2
    result[2] = -c6 * y3 + c2 * y4 + c7 * y5
    result[3] = c3 * y2 + c1 * y3 - c8 * y4
    result[4] = -c9 * y5 + c2 * y6 + c2 * y7
    result[5] = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
    result[6] = c10 * y6 * y8 - c12 * y7
    result[7] = -c10 * y6 * y8 + c12 * y7
    return result

@numba.njit(cache=True, fastmath=True)
def jacobian_numba(t, y, c):
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]
    y6, y8 = y[5], y[7]
    
    jac = np.zeros((8, 8))
    jac[0, 0] = -c1
    jac[0, 1] = c2
    jac[0, 2] = c3
    jac[1, 0] = c1
    jac[1, 1] = -c5
    jac[2, 2] = -c6
    jac[2, 3] = c2
    jac[2, 4] = c7
    jac[3, 1] = c3
    jac[3, 2] = c1
    jac[3, 3] = -c8
    jac[4, 4] = -c9
    jac[4, 5] = c2
    jac[4, 6] = c2
    jac[5, 3] = c11
    jac[5, 4] = c1
    jac[5, 5] = -c10 * y8 - c2
    jac[5, 6] = c11
    jac[5, 7] = -c10 * y6
    jac[6, 5] = c10 * y8
    jac[6, 6] = -c12
    jac[6, 7] = c10 * y6
    jac[7, 5] = -c10 * y8
    jac[7, 6] = c12
    jac[7, 7] = -c10 * y6
    return jac

# Warm upNumPy compile at import time
_dummy_y = np.zeros(8)
_dummy_c = np.zeros(12)
_ = hires_numba(0.0, _dummy_y, _dummy_c)
_ = jacobian_numba(0.0, _dummy_y, _dummy_c)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        constants = np.array(problem["constants"], dtype=np.float64)
        
        def hires_wrapper(t, y):
            return hires_numba(t, y, constants)
        
        def jacobian_wrapper(t, y):
            return jacobian_numba(t, y, constants)
        
        sol = solve_ivp(
            hires_wrapper,
            [t0, t1],
            y0,
            method="LSODA",
            rtol=1e-10,
            atol=1e-9,
            jac=jacobian_wrapper,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")