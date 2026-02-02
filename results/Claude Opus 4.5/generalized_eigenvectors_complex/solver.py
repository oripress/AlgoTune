import numpy as np
import scipy.linalg as la
from numba import njit

@njit(cache=True)
def normalize_and_prepare(eigenvectors):
    n = eigenvectors.shape[1]
    for i in range(n):
        norm = 0.0
        for j in range(eigenvectors.shape[0]):
            val = eigenvectors[j, i]
            norm += val.real * val.real + val.imag * val.imag
        norm = np.sqrt(norm)
        if norm > 1e-15:
            for j in range(eigenvectors.shape[0]):
                eigenvectors[j, i] = eigenvectors[j, i] / norm
    return eigenvectors

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        
        # Convert to standard eigenvalue problem: B^(-1)A x = λ x
        try:
            M = la.solve(B, A, check_finite=False)
            eigenvalues, eigenvectors = la.eig(M, overwrite_a=True, check_finite=False)
        except la.LinAlgError:
            scale_B = np.sqrt(la.norm(B))
            eigenvalues, eigenvectors = la.eig(A / scale_B, B / scale_B, check_finite=False)
        
        # Normalize (numba JIT)
        eigenvectors = normalize_and_prepare(eigenvectors.copy())
        
        # Sort by descending real, then descending imag
        sort_idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        sorted_eigenvalues = eigenvalues[sort_idx]
        sorted_eigenvectors = eigenvectors[:, sort_idx]
        
        return (sorted_eigenvalues.tolist(), sorted_eigenvectors.T.tolist())