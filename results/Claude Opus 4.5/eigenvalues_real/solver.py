import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute eigenvalues of a symmetric matrix.
        Returns eigenvalues in descending order.
        """
        # Use numpy's eigvalsh directly - it's optimized for symmetric matrices
        # and returns eigenvalues only
        eigenvalues = np.linalg.eigvalsh(problem)
        
        # Return in descending order (eigvalsh returns in ascending order)
        # Using negative indexing with step is faster than sort
        return eigenvalues[::-1].tolist()