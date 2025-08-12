import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        The problem is defined as: A · x = λ B · x.
        
        :param problem: Tuple (A, B) with A symmetric and B symmetric positive definite.
        :return: Tuple (eigenvalues, eigenvectors) with eigenvalues sorted in descending order.
        """
        A, B = problem
        
        # Solve generalized eigenvalue problem directly
        eigenvalues, eigenvectors = eigh(A, B)
        
        # Reverse order to have descending eigenvalues and corresponding eigenvectors
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Convert to lists more efficiently
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()  # Transpose to get row-wise eigenvectors
        
        return (eigenvalues_list, eigenvectors_list)