import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem, **kwargs):
        """Compute eigenvalues of a symmetric matrix in descending order."""
        n = problem.shape[0]
        # Use sparse eigenvalue solver for all eigenvalues
        # k='None' means compute all eigenvalues
        eigenvalues = eigsh(problem, k=n, which='LM', return_eigenvectors=False)
        # eigsh returns in ascending order for LM, so we need to reverse
        return eigenvalues[::-1].tolist()