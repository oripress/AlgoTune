import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        # Convert input to NumPy array with float64 precision
        A = np.asarray(problem, dtype=np.float64)
        
        # Use 'evd' driver (divide-and-conquer) with in-place operations
        eigenvalues, eigenvectors = eigh(
            A, 
            driver='evd', 
            overwrite_a=True,
            check_finite=False  # Skip finite check for performance
        )
        
        # Reverse for descending order using efficient slicing
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Efficient list conversion
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()  # Transpose to get row vectors
        
        return (eigenvalues_list, eigenvectors_list)