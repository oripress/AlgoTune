import numpy as np
from numpy.typing import NDArray
from typing import List
from scipy.linalg import eigvalsh

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> List[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.
        
        :param problem: A symmetric numpy matrix.
        :return: List of eigenvalues in descending order.
        """
        # Use scipy's eigvalsh function which is optimized for symmetric matrices
        # and only computes eigenvalues (not eigenvectors)
        eigenvalues = eigvalsh(problem)
        # Sort eigenvalues in descending order using numpy
        return np.sort(eigenvalues)[::-1].tolist()