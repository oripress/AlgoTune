import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Find the two eigenvalues closest to zero for a symmetric matrix.
        
        Args:
            problem (dict): Contains 'matrix', the symmetric matrix.
        
        Returns:
            list: The two eigenvalues closest to zero sorted by absolute value.
        """
        matrix = np.array(problem["matrix"], dtype=float)
        n = len(matrix)
        
        if n <= 2:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return [float(x) for x in sorted(eigenvalues, key=abs)]
        
        # Use sparse eigenvalue solver to find 2 eigenvalues closest to 0
        # sigma=0 shifts the problem to find eigenvalues near zero
        # which='LM' finds largest magnitude in shifted problem (closest to sigma)
        try:
            eigenvalues = eigsh(matrix, k=2, sigma=0, return_eigenvectors=False, which='LM')
            return [float(x) for x in sorted(eigenvalues, key=abs)]
        except:
            # Fallback to full eigenvalue decomposition if sparse solver fails
            eigenvalues = np.linalg.eigvalsh(matrix)
            abs_eigenvalues = np.abs(eigenvalues)
            indices = np.argpartition(abs_eigenvalues, 1)[:2]
            two_smallest = eigenvalues[indices]
            return [float(x) for x in sorted(two_smallest, key=abs)]