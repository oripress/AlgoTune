import numpy as np
import scipy.linalg.blas as _blas
import os

# Try to maximize BLAS thread usage
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

_dgemm = getattr(_blas, 'dgemm')
_sgemm = getattr(_blas, 'sgemm')

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["A"]
        B = problem["B"]
        
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64)
        elif A.dtype != np.float64:
            A = A.astype(np.float64)
            
        if not isinstance(B, np.ndarray):
            B = np.asarray(B, dtype=np.float64)
        elif B.dtype != np.float64:
            B = B.astype(np.float64)
        
        n, m = A.shape
        p = B.shape[1]
        
        # Use dgemm with transpose trick for C-contiguous arrays
        if A.flags['C_CONTIGUOUS'] and B.flags['C_CONTIGUOUS']:
            return _dgemm(1.0, B.T, A.T).T
        elif A.flags['F_CONTIGUOUS'] and B.flags['F_CONTIGUOUS']:
            return _dgemm(1.0, A, B)
        else:
            return np.dot(A, B)