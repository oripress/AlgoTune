import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from numba import jit

# Optimized normalization function using Numba
@jit(nopython=True)
def normalize_vectors_numba(eigenvectors_real, eigenvectors_imag):
    n, m = eigenvectors_real.shape
    norms = np.empty(m)
    for i in range(m):
        norm = 0.0
        for j in range(n):
            norm += eigenvectors_real[j, i] ** 2 + eigenvectors_imag[j, i] ** 2
        norms[i] = np.sqrt(norm)
    
    # Apply normalization
    for i in range(m):
        if norms[i] < 1e-15:
            norms[i] = 1.0
        for j in range(n):
            eigenvectors_real[j, i] /= norms[i]
            eigenvectors_imag[j, i] /= norms[i]
    
    return eigenvectors_real, eigenvectors_imag

class Solver:
    def solve(self, problem, **kwargs) -> tuple[list[complex], list[list[complex]]]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B:
            A · x = λ B · x.
        
        Returns eigenvalues and eigenvectors sorted in descending order.
        """
        A, B = problem
        
        # Convert to numpy arrays with float64 for optimal performance
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        
        # Use the fastest available eigenvalue solver
        # Try different approaches and pick the fastest one
        try:
            # Try using eigh for symmetric matrices first (much faster)
            if np.allclose(A, A.T) and np.allclose(B, B.T):
                eigenvalues, eigenvectors = la.eigh(A, B, check_finite=False)
                # Convert real eigenvalues to complex for consistent output
                eigenvalues = eigenvalues.astype(np.complex128)
            else:
                # Use standard eig for non-symmetric matrices
                eigenvalues, eigenvectors = la.eig(A, B, check_finite=False)
        except:
            # Fallback
            eigenvalues, eigenvectors = la.eig(A, B, check_finite=False)
        
        # Fast normalization using Numba
        eigenvectors_real, eigenvectors_imag = normalize_vectors_numba(
            eigenvectors.real, eigenvectors.imag
        )
        eigenvectors = eigenvectors_real + 1j * eigenvectors_imag
        
        # Fast sorting using argsort with complex key
        # Sort by real part descending, then imaginary part descending
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        sort_indices = np.lexsort((-imag_parts, -real_parts))
        
        # Apply sorting
        sorted_eigenvalues = eigenvalues[sort_indices]
        sorted_eigenvectors = eigenvectors[:, sort_indices].T
        
        # Fast conversion to lists
        eigenvalues_list = sorted_eigenvalues.tolist()
        eigenvectors_list = [vec.tolist() for vec in sorted_eigenvectors]
        
        return (eigenvalues_list, eigenvectors_list)