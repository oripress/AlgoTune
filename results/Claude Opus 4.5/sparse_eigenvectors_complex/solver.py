from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        
        # For small/medium matrices, dense eigenvalue decomposition is often faster
        threshold = 300
        if N <= threshold:
            if sparse.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = np.asarray(A)
            
            eigenvalues, eigenvectors = np.linalg.eig(A_dense)
            
            # Get indices of k largest eigenvalues by modulus
            indices = np.argsort(-np.abs(eigenvalues))[:k]
            
            solution = [eigenvectors[:, i] for i in indices]
            return solution
        
        # For larger matrices, use sparse eigenvalue solver with optimized params
        v0 = np.ones(N, dtype=A.dtype)
        
        eigenvalues, eigenvectors = eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )
        
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda pair: -np.abs(pair[0]))
        
        solution = [pair[1] for pair in pairs]
        
        return solution