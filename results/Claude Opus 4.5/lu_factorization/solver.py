import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute LU factorization of matrix A.
        A = P · L · U
        """
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64)
        elif A.dtype != np.float64:
            A = A.astype(np.float64)
        
        # Use check_finite=False to skip validation for speed
        P, L, U = lu(A, check_finite=False, overwrite_a=True)
        return {"LU": {"P": P, "L": L, "U": U}}