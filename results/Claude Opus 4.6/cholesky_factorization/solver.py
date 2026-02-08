import numpy as np
import scipy.linalg as la

class Solver:
    def __init__(self):
        self._dpotrf = la.get_lapack_funcs('potrf', (np.array([[1.0]]),))
    
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64, order='F')
        elif A.dtype != np.float64:
            A = np.array(A, dtype=np.float64, order='F')
        elif not A.flags['F_CONTIGUOUS']:
            # For symmetric matrix, A.T has same values but is F-contiguous
            # This avoids a copy
            A = A.T
        
        L, info = self._dpotrf(A, lower=True, overwrite_a=True, clean=True)
        return {"Cholesky": {"L": L}}