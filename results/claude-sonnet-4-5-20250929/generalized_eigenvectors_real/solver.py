import numpy as np
from scipy import linalg
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x efficiently.
        
        Uses scipy.linalg.eigh directly with b parameter for optimal performance.
        
        :param problem: Tuple (A, B) with A symmetric and B symmetric positive definite.
        :return: Tuple (eigenvalues, eigenvectors) with eigenvalues sorted in descending order.
        """
        A, B = problem
        
        # Use scipy's generalized eigenvalue solver directly
        eigenvalues, eigenvectors = linalg.eigh(A, b=B, check_finite=False, overwrite_a=True, overwrite_b=True)
        
        # Reverse order for descending eigenvalues
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Convert to lists efficiently
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()
        
        return (eigenvalues_list, eigenvectors_list)