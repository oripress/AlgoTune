from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[np.ndarray]:
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        
        # For small-to-medium matrices, dense eigensolve is often faster
        if N <= 1000:
            A_dense = A.toarray()
            eigenvalues, eigenvectors = np.linalg.eig(A_dense)
            idx = np.argsort(-np.abs(eigenvalues))[:k]
            return [eigenvectors[:, i] for i in idx]
        
        # For larger matrices, use sparse eigensolver with optimized parameters
        v0 = np.ones(N, dtype=A.dtype)
        ncv = min(max(2 * k + 1, 20), N - 1)
        eigenvalues, eigenvectors = eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=ncv,
        )
        
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda pair: -np.abs(pair[0]))
        return [pair[1] for pair in pairs]