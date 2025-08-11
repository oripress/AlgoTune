import numpy as np
from scipy import linalg
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.
        """
        # Use eigvalsh for symmetric matrices - only computes eigenvalues, not eigenvectors
        eigenvalues = linalg.eigvalsh(problem)
        # eigvalsh returns values in ascending order, so reverse them
        return eigenvalues[::-1].tolist()