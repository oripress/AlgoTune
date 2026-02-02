import numpy as np
from scipy.integrate import ode, solve_ivp
from numba import njit
from typing import Any

@njit(fastmath=True)
def hires_func(t, y, c):
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = c
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    
    dy1 = -c1*y1 + c2*y2 + c3*y3 + c4
    dy2 = c1*y1 - c5*y2
    dy3 = -c6*y3 + c2*y4 + c7*y5
    dy4 = c3*y2 + c1*y3 - c8*y4
    dy5 = -c9*y5 + c2*y6 + c2*y7
    dy6 = -c10*y6*y8 + c11*y4 + c1*y5 - c2*y6 + c11*y7
    dy7 = c10*y6*y8 - c12*y7
    dy8 = -c10*y6*y8 + c12*y7
    
    return np.array([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8])

@njit(fastmath=True)
def hires_jac(t, y, c):
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = c
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    
    J = np.zeros((8, 8))
    
    J[0, 0] = -c1
    J[0, 1] = c2
    J[0, 2] = c3
    
    J[1, 0] = c1
    J[1, 1] = -c5
    
    J[2, 2] = -c6
    J[2, 3] = c2
    J[2, 4] = c7
    
    J[3, 1] = c3
    J[3, 2] = c1
    J[3, 3] = -c8
    
    J[4, 4] = -c9
    J[4, 5] = c2
    J[4, 6] = c2
    
    J[5, 3] = c11
    J[5, 4] = c1
    J[5, 5] = -c10*y8 - c2
    J[5, 6] = c11
    J[5, 7] = -c10*y6
    
    J[6, 5] = c10*y8
    J[6, 6] = -c12
    J[6, 7] = c10*y6
    
    J[7, 5] = -c10*y8
    J[7, 6] = c12
    J[7, 7] = -c10*y6
    
    return J

class Solver:
    def __init__(self):
        # Trigger compilation
        dummy_y = np.zeros(8)
        dummy_c = np.ones(12)
        hires_func(0.0, dummy_y, dummy_c)
        hires_jac(0.0, dummy_y, dummy_c)

    def solve(self, problem, **kwargs) -> Any:
        y0 = np.array(problem["y0"])
        t0 = problem["t0"]
        t1 = problem["t1"]
        constants = np.array(problem["constants"])
        
        # Wrapper to ensure arguments are passed correctly
        def func_wrapper(t, y, c):
            return hires_func(t, y, c)
            
        def jac_wrapper(t, y, c):
            return hires_jac(t, y, c)

        # solve_ivp with LSODA is generally very fast for stiff systems
        sol = solve_ivp(
            func_wrapper,
            (t0, t1),
            y0,
            method='LSODA',
            jac=jac_wrapper,
            args=(constants,),
            rtol=1e-10,
            atol=1e-9
        )
        
        if not sol.success:
             raise RuntimeError(f"Integration failed: {sol.message}")

        return sol.y[:, -1].tolist()