import numpy as np
from numpy.typing import NDArray
import scipy.linalg

class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.
        """
        # Use scipy's eigh which is optimized for symmetric matrices
        # eigvals_only=True skips eigenvector computation for speed
        eigenvalues = scipy.linalg.eigh(problem, eigvals_only=True)
        
        # Sort eigenvalues in descending order using numpy's sort which is faster
        return np.sort(eigenvalues)[::-1].tolist()