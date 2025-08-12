import numpy as np
from scipy import linalg
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        The problem is defined as: A · x = λ B · x.
        """
        A, B = problem
        
        # Use scipy.linalg.eigh with optimized parameters
        # check_finite=False skips checks since we trust the input
        # overwrite_a=False, overwrite_b=False to avoid modifying input
        eigenvalues, eigenvectors = linalg.eigh(A, B, type=1, check_finite=False, 
                                                overwrite_a=False, overwrite_b=False)
        
        # Reverse order to have descending eigenvalues and corresponding eigenvectors
        # Use flip which might be more efficient than slicing
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.flip(eigenvectors, axis=1)
        
        # More efficient conversion to lists
        eigenvalues_list = eigenvalues.tolist()
        # Use transpose and tolist() for better memory access pattern
        eigenvectors_list = eigenvectors.T.tolist()
        
        return (eigenvalues_list, eigenvectors_list)