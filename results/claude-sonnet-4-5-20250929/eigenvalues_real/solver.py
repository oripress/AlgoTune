import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.
        
        :param problem: A symmetric numpy matrix.
        :return: List of eigenvalues in descending order.
        """
        # Use numpy's eigvalsh - only computes eigenvalues
        eigenvalues = np.linalg.eigvalsh(problem)
        # eigvalsh returns in ascending order, reverse for descending
        return eigenvalues[::-1].tolist()