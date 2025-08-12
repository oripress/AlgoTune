import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        """
        Solve the eigenvalue problem for the given real symmetric matrix.
        The solution returned is a tuple (eigenvalues, eigenvectors) where:
          - eigenvalues is a list of floats sorted in descending order.
          - eigenvectors is a list of lists, where each inner list represents the corresponding
            eigenvector (normalized to have unit length), sorted corresponding to the eigenvalues.

        :param problem: A numpy array representing the real symmetric matrix.
        :return: Tuple (eigenvalues, eigenvectors)
        """
        # Use scipy.linalg.eigh with optimized parameters
        eigenvalues, eigenvectors = scipy.linalg.eigh(problem, check_finite=False, overwrite_a=False, overwrite_b=False, driver='evd')
        # Reverse order to have descending eigenvalues and corresponding eigenvectors.
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        # Convert eigenvalues to a list and eigenvectors to a list of lists.
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = [eigenvectors[:, i].tolist() for i in range(eigenvectors.shape[1])]
        return (eigenvalues_list, eigenvectors_list)