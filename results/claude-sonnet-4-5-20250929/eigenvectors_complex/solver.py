import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[list[complex]]:
        """
        Fully vectorized solver using np.linalg.eig.
        """
        eigenvalues, eigenvectors = np.linalg.eig(problem)
        
        # Sort using argsort with key tuple
        sort_idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        # Reorder columns
        eigenvectors = eigenvectors[:, sort_idx]
        
        # Vectorized normalization
        norms = np.linalg.norm(eigenvectors, axis=0)
        eigenvectors = eigenvectors / np.where(norms > 1e-12, norms, 1.0)
        
        # Fast conversion to list
        return eigenvectors.T.tolist()