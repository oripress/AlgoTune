import numpy as np
import scipy.linalg
import numba as nb

@nb.njit(fastmath=True, cache=True)
def sqrtm_2x2(A):
    """Fast 2x2 matrix square root"""
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    tau = a + d
    delta = a*d - b*c
    s = np.sqrt(tau + 2*np.sqrt(delta))
    if abs(s) < 1e-12:
        return np.zeros_like(A)
    X = np.empty_like(A)
    X[0,0] = (a + np.sqrt(delta))/s
    X[0,1] = b/s
    X[1,0] = c/s
    X[1,1] = (d + np.sqrt(delta))/s
    return X

class Solver:
    def __init__(self):
        # Warm up numba
        dummy = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=np.complex128)
        sqrtm_2x2(dummy)
    
    def solve(self, problem: dict) -> dict:
        A = np.asarray(problem["matrix"], dtype=np.complex128)
        n = A.shape[0]
        
        try:
            if n == 2:
                X = sqrtm_2x2(A)
            else:
                X = scipy.linalg.sqrtm(A, disp=False)[0]
            return {"sqrtm": {"X": X.tolist()}}
        except:
            return {"sqrtm": {"X": []}}