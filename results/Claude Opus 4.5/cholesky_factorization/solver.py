import numpy as np
from scipy.linalg import cholesky as scipy_cholesky

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        # Convert to contiguous array with correct dtype for optimal LAPACK
        if not isinstance(A, np.ndarray):
            A = np.ascontiguousarray(A, dtype=np.float64)
        elif not A.flags['C_CONTIGUOUS'] or A.dtype != np.float64:
            A = np.ascontiguousarray(A, dtype=np.float64)
        
        # scipy cholesky with check_finite=False skips input validation
        return {"Cholesky": {"L": scipy_cholesky(A, lower=True, check_finite=False)}}