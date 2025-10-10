from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[np.ndarray]:
        """
        Compute eigenvectors for the largest k eigenvalues of a sparse matrix.
        
        :param problem: Dictionary with 'matrix' (sparse CSR) and 'k' (number of eigenvalues)
        :return: List of eigenvectors sorted by descending eigenvalue modulus
        """
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        v0 = np.ones(N, dtype=A.dtype)
        
        # Optimize ncv for better performance
        ncv = min(max(2 * k + 1, 20), N)
        
        # Compute eigenvalues and eigenvectors with optimized parameters
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 100,  # Reduce iterations for speed
            ncv=ncv,
            tol=0,  # Keep default tolerance for accuracy
        )
        
        # Create pairs and sort - matching reference implementation exactly
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda pair: -np.abs(pair[0]))
        
        # Extract eigenvectors
        return [pair[1] for pair in pairs]