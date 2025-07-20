from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square sparse matrix.
        Returns eigenvectors with the largest k eigenvalues sorted in descending order by modulus.
        """
        A = problem["matrix"]
        k = problem["k"]
        
        # Convert to sparse matrix if needed
        if isinstance(A, (list, np.ndarray)):
            A = sparse.csr_matrix(A)
            
        N = A.shape[0]
        
        # If k >= N - 1, we need to use dense eigenvalue computation
        if k >= N - 1:
            # Convert to dense array and use regular eig
            A_dense = A.toarray() if sparse.issparse(A) else np.array(A)
            eigenvalues, eigenvectors = np.linalg.eig(A_dense)
            # Select all eigenvalues and sort by modulus
            pairs = list(zip(eigenvalues, eigenvectors.T))
            pairs.sort(key=lambda pair: -np.abs(pair[0]))
            solution = [pair[1] for pair in pairs[:k]]
        else:
            # Create a deterministic starting vector
            v0 = np.ones(N, dtype=A.dtype)
            
            # Compute eigenvalues using sparse.linalg.eigs
            eigenvalues, eigenvectors = sparse.linalg.eigs(
                A,
                k=k,
                v0=v0,
                maxiter=N * 200,
                ncv=max(2 * k + 1, 20),
            )
            
            # Sort by descending order of eigenvalue modulus
            pairs = list(zip(eigenvalues, eigenvectors.T))
            pairs.sort(key=lambda pair: -np.abs(pair[0]))
            solution = [pair[1] for pair in pairs]
            
        return solution