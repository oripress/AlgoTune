import numpy as np
from scipy.linalg import svd

class Solver:
    def solve(self, problem, **kwargs):
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}
        
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        
        if A.shape != B.shape:
            return {}
        
        # Compute M = B @ A^T
        M = B @ A.T
        
        # Use SVD
        U, _, Vt = svd(M, full_matrices=False, lapack_driver='gesdd', check_finite=False, overwrite_a=True)
        
        # Compute G = U @ V^T
        G = U @ Vt
        
        return {"solution": G}