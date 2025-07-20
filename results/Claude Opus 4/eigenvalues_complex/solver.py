import numpy as np
from numpy.typing import NDArray
from typing import Any
import scipy.linalg

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square matrix.
        The solution returned is a list of eigenvalues sorted in descending order.
        The sorting order is defined as follows: first by the real part (descending),
        then by the imaginary part (descending).
        
        :param problem: A numpy array representing the real square matrix.
        :return: List of eigenvalues (complex numbers) sorted in descending order.
        """
        # Compute only eigenvalues (not eigenvectors) using scipy
        eigenvalues = scipy.linalg.eigvals(problem)
        # Sort eigenvalues: descending order by real part, then by imaginary part
        solution = sorted(eigenvalues, key=lambda x: (-x.real, -x.imag))
        return solution