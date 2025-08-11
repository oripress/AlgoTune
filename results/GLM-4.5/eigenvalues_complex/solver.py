import numpy as np
from numpy.typing import NDArray
from scipy import linalg

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
        # Compute eigenvalues using scipy.linalg.eigvals with optimizations
        eigenvalues = linalg.eigvals(problem, check_finite=False)
        
        # Optimized sorting using NumPy operations
        # Create negative values for descending order
        neg_real = -eigenvalues.real
        neg_imag = -eigenvalues.imag
        
        # Use lexsort for efficient multi-key sorting
        # lexsort sorts by last key first, so we put imag last
        sorted_indices = np.lexsort((neg_imag, neg_real))
        
        # Apply sorting and convert to list
        return eigenvalues[sorted_indices].tolist()