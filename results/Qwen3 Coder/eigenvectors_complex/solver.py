import numpy as np
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Compute eigenvalues and eigenvectors of a real square matrix.
        
        Returns eigenvalues and eigenvectors sorted by eigenvalue (real part first,
        then imaginary part) in descending order. Eigenvectors are normalized.
        """
        # Compute eigenvalues and eigenvectors using numpy
        eigenvalues, eigenvectors = np.linalg.eig(problem)
        
        # Sort by descending order of eigenvalue real part, then imaginary part
        sort_indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        # Extract and sort eigenvectors in one step
        sorted_evec = eigenvectors[:, sort_indices]
        
        # Normalize using numpy's built-in function for better performance
        # Using reciprocal for potentially better performance
        norms = np.linalg.norm(sorted_evec, axis=0)
        normalized_evec = sorted_evec * (1.0 / norms)
        
        # Convert to list of lists using numpy's tolist method on the transposed array
        return normalized_evec.T.tolist()