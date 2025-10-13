from typing import Any, List
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> List[complex]:
        """
        Compute and return the eigenvalues of a real square matrix, sorted in descending order:
        first by real part (descending), then by imaginary part (descending).
        """
        A = np.asarray(problem)
        # Compute eigenvalues only (faster than eig which also computes eigenvectors)
        eigvals = np.linalg.eigvals(A)

        if eigvals.size == 0:
            return []

        # Sort by real part desc, then imaginary part desc using lexsort
        re = eigvals.real
        im = eigvals.imag
        order = np.lexsort((-im, -re))
        sorted_eigs = eigvals[order]

        # Convert to Python complex numbers
        return sorted_eigs.tolist()