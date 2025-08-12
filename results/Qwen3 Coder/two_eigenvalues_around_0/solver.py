import numpy as np
from typing import Any, Dict, List
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]], **kwargs) -> List[float]:
        """
        Solve the problem by finding the two eigenvalues closest to zero.
        
        matrix = problem["matrix"]
            problem (dict): Contains 'matrix', the symmetric matrix.
            
        Returns:
            list: The two eigenvalues closest to zero sorted by absolute value.
        """
        # Convert to numpy array
        matrix = np.array(problem["matrix"], dtype=np.float64)
        
        # For small matrices, direct computation is faster
        if matrix.shape[0] <= 100:
            eigenvalues = np.linalg.eigvalsh(matrix)
            
            # Find the two eigenvalues closest to zero using argpartition
            abs_eigenvalues = np.abs(eigenvalues)
            indices = np.argpartition(abs_eigenvalues, min(1, len(abs_eigenvalues)))[:2]
            
            # Get the two closest eigenvalues
            result = eigenvalues[indices]
        else:
            # For larger matrices, use scipy's eigsh with shift-invert mode
            try:
                # Use scipy's eigsh with shift-invert mode to find eigenvalues closest to 0
                eigenvalues = eigsh(matrix, k=min(2, matrix.shape[0]), sigma=0.0, which='LM', return_eigenvectors=False)
                result = eigenvalues
            except:
                # Fallback to standard method if scipy method fails
                eigenvalues = np.linalg.eigvalsh(matrix)
                abs_eigenvalues = np.abs(eigenvalues)
                indices = np.argpartition(abs_eigenvalues, min(1, len(abs_eigenvalues)))[:2]
                result = eigenvalues[indices]
        
        # Sort by absolute value
        return sorted(result.tolist(), key=abs)