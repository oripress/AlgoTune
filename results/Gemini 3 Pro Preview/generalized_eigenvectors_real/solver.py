import numpy as np
try:
    from fast_solver_float import solve_cython_float
except ImportError:
    solve_cython_float = None
from scipy.linalg.lapack import ssygvd

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        
        if solve_cython_float:
            # Ensure Fortran-contiguous float32 arrays for Cython
            A_arr = np.array(A, dtype=np.float32, order='F')
            B_arr = np.array(B, dtype=np.float32, order='F')
            return solve_cython_float(A_arr, B_arr)
        else:
            # Fallback
            A = np.array(A, dtype=np.float32, order='F')
            B = np.array(B, dtype=np.float32, order='F')
            w, v, info = ssygvd(A, B, itype=1, jobz='V', uplo='L', overwrite_a=True, overwrite_b=True)
            w = w[::-1]
            v = v[:, ::-1]
            return w.tolist(), v.T.tolist()