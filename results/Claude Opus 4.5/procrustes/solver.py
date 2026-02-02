import numpy as np
from scipy.linalg import svd

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the orthogonal Procrustes problem.
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}
        
        # Convert to numpy arrays if not already
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64)
        if not isinstance(B, np.ndarray):
            B = np.array(B, dtype=np.float64)
        
        if A.shape != B.shape:
            return {}
        
        # Compute M = B A^T
        M = B @ A.T
        
        # Compute SVD of M using scipy with gesdd driver (faster)
        U, _, Vt = svd(M, full_matrices=False, check_finite=False, lapack_driver='gesdd')
        
        # Compute G = U V^T
        G = U @ Vt
        
        return {"solution": G}