import numpy as np
from scipy import linalg

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        
        The problem is defined as: A · x = λ B · x.
        Uses scipy.linalg.eigh with optimizations.
        
        :param problem: Tuple (A, B) where A is symmetric and B is symmetric positive definite.
        :return: List of eigenvalues sorted in descending order.
        """
        A, B = problem
        
        # Use scipy's generalized eigenvalue solver with additional optimizations
        # overwrite_a=True and overwrite_b=True to save memory (though we can't use here as we need originals)
        # check_finite=False to skip finite check for speed
        eigenvalues = linalg.eigh(A, B, eigvals_only=True, check_finite=False)
        
        # Use numpy's sort which is faster than Python's sorted for numpy arrays
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues.tolist()