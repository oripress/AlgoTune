import numpy as np
from numpy.typing import NDArray
from scipy import linalg

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """
        Compute eigenvalues of a square matrix and return them sorted in descending order.
        Sorting: first by real part (descending), then by imaginary part (descending).
        """
        # Compute eigenvalues only (faster than computing eigenvectors too)
        eigenvalues = np.linalg.eigvals(problem)
        # Sort eigenvalues in descending order using numpy's lexsort (faster than sorted with lambda)
        # lexsort sorts by the last key first, so we negate both to get descending order
        indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        return eigenvalues[indices].tolist()