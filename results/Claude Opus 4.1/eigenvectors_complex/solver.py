import numpy as np
from scipy import linalg

class Solver:
    def solve(self, problem):
        """
        Solve the eigenvector problem using scipy.linalg.eig which can be faster.
        """
        # Use scipy's eig which can be faster for certain matrices
        eigenvalues, eigenvectors = linalg.eig(problem, check_finite=False)
        
        # Create sorting indices - use argsort with custom key
        sort_indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        # Direct column indexing is faster than iterating
        sorted_eigenvectors = eigenvectors[:, sort_indices]
        
        # Convert to list in one go - eigenvectors from eig are already normalized
        # We can skip normalization check for most cases
        return [sorted_eigenvectors[:, i].tolist() for i in range(sorted_eigenvectors.shape[1])]