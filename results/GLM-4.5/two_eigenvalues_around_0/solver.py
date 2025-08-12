import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem, **kwargs) -> list[float]:
        """
        Solve the problem by finding the two eigenvalues closest to zero.

        Args:
            problem (dict): Contains 'matrix', the symmetric matrix.

        Returns:
            list: The two eigenvalues closest to zero sorted by absolute value.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        
        # Try sparse eigsh approach with optimized parameters
        try:
            # Use more optimized parameters for eigsh
            eigenvalues = eigsh(matrix, k=2, sigma=0, which='LM', return_eigenvectors=False, 
                              tol=1e-6, maxiter=1000)
            eigenvalues_sorted = sorted(eigenvalues, key=abs)
            return [float(eigenvalues_sorted[0]), float(eigenvalues_sorted[1])]
        except Exception:
            # Fallback to full numpy computation
            eigenvalues = np.linalg.eigvalsh(matrix)
            eigenvalues_sorted = sorted(eigenvalues, key=abs)
            return [float(eigenvalues_sorted[0]), float(eigenvalues_sorted[1])]