import numpy as np
from numba import jit
from typing import List

@jit(nopython=True)
def normalize_eigenvectors(eigenvectors):
    """Numba-compiled function to normalize eigenvectors efficiently."""
    n = eigenvectors.shape[0]
    m = eigenvectors.shape[1]
    normalized = np.empty_like(eigenvectors)
    
    for i in range(n):
        # Compute norm
        norm = 0.0
        for j in range(m):
            real_part = eigenvectors[i, j].real
            imag_part = eigenvectors[i, j].imag
            norm += real_part * real_part + imag_part * imag_part
        
        if norm > 1e-24:
            norm = np.sqrt(norm)
            # Normalize
            for j in range(m):
                normalized[i, j] = eigenvectors[i, j] / norm
        else:
            # Keep as is if norm is too small
            for j in range(m):
                normalized[i, j] = eigenvectors[i, j]
    
    return normalized

class Solver:
    def solve(self, problem: List[List[float]], **kwargs) -> List[List[complex]]:
        """
        Solve the eigenvector problem for the given non-symmetric matrix.
        Compute eigenvalues and eigenvectors using np.linalg.eig.
        Sort the eigenpairs in descending order by the real part (and then imaginary part) of the eigenvalues.
        Return the eigenvectors (each normalized to unit norm) as a list of lists of complex numbers.
        
        :param problem: A non-symmetric square matrix represented as a list of lists of real numbers.
        :return: A list of normalized eigenvectors sorted in descending order.
        """
        # Convert input to numpy array
        A = np.array(problem, dtype=np.float64)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Get sorting indices based on eigenvalue real and imaginary parts
        # Using negative for descending order
        sort_indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        # Apply sorting to eigenvectors (already transposed)
        sorted_evecs = eigenvectors.T[sort_indices]
        
        # Normalize using numba-compiled function
        normalized_evecs = normalize_eigenvectors(sorted_evecs)
        
        return normalized_evecs.tolist()