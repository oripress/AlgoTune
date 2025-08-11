import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square matrix.
        The solution returned is a list of eigenvalues sorted in descending order.
        The sorting order is defined as follows: first by the real part (descending),
        then by the imaginary part (descending).
        
        :param problem: A numpy array representing the real square matrix.
        :return: List of eigenvalues (complex numbers) sorted in descending order.
        """
        # Use eigvals for eigenvalues only (faster than eig)
        eigenvalues = np.linalg.eigvals(problem)
        
        # Use numpy's lexsort for faster sorting - avoid lambda overhead
        # lexsort sorts in ascending order, so negate for descending
        indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        # Return as list
        return eigenvalues[indices].tolist()