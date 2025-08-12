import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs) -> list[float]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        
        The problem is defined as: A · x = λ B · x.
        Uses scipy.linalg.eigh with the 'gvx' driver which is currently the fastest.
        
        :param problem: Tuple (A, B) where A is symmetric and B is symmetric positive definite.
        :return: List of eigenvalues sorted in descending order.
        """
        A, B = problem
        
        # Convert to numpy arrays with optimal memory layout
        A = np.asarray(A, dtype=np.float64, order='C')
        B = np.asarray(B, dtype=np.float64, order='C')
        
        # Use scipy.linalg.eigh with the 'gvx' driver
        # This driver is designed for generalized eigenvalue problems
        eigenvalues = eigh(A, B, eigvals_only=True, driver='gvx', check_finite=False)
        
        # Reverse to get descending order
        solution = eigenvalues[::-1].tolist()
        return solution