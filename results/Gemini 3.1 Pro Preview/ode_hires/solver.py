import numpy as np
from scipy.integrate import odeint
from numba import njit

@njit
def hires_numba(y, t, c):
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    f1 = -c[0] * y1 + c[1] * y2 + c[2] * y3 + c[3]
    f2 = c[0] * y1 - c[4] * y2
    f3 = -c[5] * y3 + c[1] * y4 + c[6] * y5
    f4 = c[2] * y2 + c[0] * y3 - c[7] * y4
    f5 = -c[8] * y5 + c[1] * y6 + c[1] * y7
    f6 = -c[9] * y6 * y8 + c[10] * y4 + c[0] * y5 - c[1] * y6 + c[10] * y7
    f7 = c[9] * y6 * y8 - c[11] * y7
    f8 = -c[9] * y6 * y8 + c[11] * y7
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

@njit
def jac_numba(y, t, c):
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    J = np.zeros((8, 8))
    J[0, 0] = -c[0]
    J[0, 1] = c[1]
    J[0, 2] = c[2]
    J[1, 0] = c[0]
    J[1, 1] = -c[4]
    J[2, 2] = -c[5]
    J[2, 3] = c[1]
    J[2, 4] = c[6]
    J[3, 1] = c[2]
    J[3, 2] = c[0]
    J[3, 3] = -c[7]
    J[4, 4] = -c[8]
    J[4, 5] = c[1]
    J[4, 6] = c[1]
    J[5, 3] = c[10]
    J[5, 4] = c[0]
    J[5, 5] = -c[9] * y8 - c[1]
    J[5, 6] = c[10]
    J[5, 7] = -c[9] * y6
    J[6, 5] = c[9] * y8
    J[6, 6] = -c[11]
    J[6, 7] = c[9] * y6
    J[7, 5] = -c[9] * y8
    J[7, 6] = c[11]
    J[7, 7] = -c[9] * y6
    return J

class Solver:
    def __init__(self):
        dummy_y = np.zeros(8)
        dummy_c = np.zeros(12)
        hires_numba(dummy_y, 0.0, dummy_c)
        jac_numba(dummy_y, 0.0, dummy_c)

    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        constants = np.array(problem["constants"])
        
        sol = odeint(
            hires_numba, 
            y0, 
            [t0, t1], 
            args=(constants,), 
            Dfun=jac_numba, 
            rtol=1e-10, 
            atol=1e-9, 
            tfirst=False,
            mxstep=5000
        )
            
        return sol[-1].tolist()