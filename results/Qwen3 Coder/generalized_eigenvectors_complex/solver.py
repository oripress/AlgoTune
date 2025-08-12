import numpy as np
import scipy.linalg as la
from typing import Any, Tuple, List

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], *args, **kwargs) -> Any:
        """Solve the generalized eigenvalue problem A · x = λ B · x"""
        A, B = problem
        
        # Direct computation with no extra overhead
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(B, A))
        
        # Ultra-fast normalization in place
        norms = np.linalg.norm(eigenvectors, axis=0)
        # Avoid division by zero
        valid = norms > 1e-15
        eigenvectors[:, valid] /= norms[valid]
        
        # Sort by real part (descending) then imaginary part (descending) 
        # Using a faster sorting approach
        sort_key = -eigenvalues.real - 1j * eigenvalues.imag
        sort_indices = np.argsort(sort_key, kind='quicksort')
        
        # Apply sorting
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Return as lists
        return (eigenvalues.tolist(), eigenvectors.T.tolist())
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Efficient conversion to lists
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()
        
        return (eigenvalues_list, eigenvectors_list)