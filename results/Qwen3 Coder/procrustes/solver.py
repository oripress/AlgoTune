import numpy as np
from scipy.linalg import svd

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        # Extract matrices A and B 
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        
        # Compute M = B @ A^T 
        M = np.dot(B, A.T)
        
        # Compute SVD of M using scipy
        U, s, Vt = svd(M, full_matrices=True, overwrite_a=True, check_finite=False)
        
        # Compute G = U @ Vt
        G = np.dot(U, Vt)
        
        return {"solution": G}