import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves a given positive semidefinite cone projection problem.
        
        The algorithm:
        1. Compute eigendecomposition of symmetric matrix A
        2. Set negative eigenvalues to zero
        3. Reconstruct the matrix X from modified eigenvalues
        """
        A = np.array(problem["A"], dtype=np.float64)
        
        # Use eigh for symmetric matrices (faster than eig)
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Clip negative eigenvalues to zero
        eigvals = np.maximum(eigvals, 0)
        
        # Reconstruct the matrix efficiently
        # X = V @ diag(lambda) @ V.T = V @ (diag(lambda) @ V.T) = V @ (lambda * V.T)
        X = eigvecs @ (eigvals[:, np.newaxis] * eigvecs.T)
        
        return {"X": X.tolist()}