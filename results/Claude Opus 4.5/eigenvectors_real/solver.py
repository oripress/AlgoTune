import numpy as np
from scipy.linalg import eigh as scipy_eigh

class Solver:
    def solve(self, problem, **kwargs):
        # Convert to numpy array with contiguous memory
        A = np.ascontiguousarray(problem, dtype=np.float64)
        n = A.shape[0]
        
        # Use divide-and-conquer with lower triangular (often slightly faster)
        eigenvalues, eigenvectors = scipy_eigh(A, lower=True, driver='evd', 
                                                check_finite=False, overwrite_a=True)
        
        # Efficient reverse - use np.flip for contiguous result
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.flip(eigenvectors, axis=1)
        
        # Direct list conversion
        return (eigenvalues.tolist(), eigenvectors.T.tolist())