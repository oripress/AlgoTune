import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Find the two eigenvalues closest to zero in a symmetric matrix.
        
        Args:
            problem: Dictionary containing 'matrix' key with symmetric matrix.
            
        Returns:
            List of two eigenvalues closest to zero, sorted by absolute value.
        """
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        n = matrix.shape[0]
        
        # For small matrices, use regular eigvalsh
        if n <= 4:
            eigenvalues = np.linalg.eigvalsh(matrix)
            abs_vals = np.abs(eigenvalues)
            indices = np.argpartition(abs_vals, min(1, n-1))[:min(2, n)]
            result = eigenvalues[indices]
            if len(result) == 2 and abs(result[0]) > abs(result[1]):
                result = result[[1, 0]]
            return result.tolist()
        
        try:
            # For larger matrices, use sparse solver targeting eigenvalues near 0
            # This is much faster when we only need a few eigenvalues
            eigenvalues = eigsh(matrix, k=min(2, n-1), sigma=0.0, which='LM', 
                               return_eigenvectors=False, tol=1e-8)
            
            # Sort by absolute value
            abs_vals = np.abs(eigenvalues)
            sorted_indices = np.argsort(abs_vals)
            return eigenvalues[sorted_indices].tolist()
            
        except:
            # Fallback to regular method if sparse solver fails
            eigenvalues = np.linalg.eigvalsh(matrix)
            abs_vals = np.abs(eigenvalues)
            indices = np.argpartition(abs_vals, 1)[:2]
            result = eigenvalues[indices]
            if abs(result[0]) > abs(result[1]):
                result = result[[1, 0]]
            return result.tolist()