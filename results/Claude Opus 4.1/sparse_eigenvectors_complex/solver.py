from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square sparse matrix.
        The solution returned is a list of the eigenvectors with the largest `k` eigenvalues 
        sorted in descending order by their modulus.
        """
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        
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