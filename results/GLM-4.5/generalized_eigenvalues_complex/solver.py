import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from numba import jit

class Solver:
    def solve(self, problem, **kwargs) -> list[complex]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.

        The problem is defined as: A · x = λ B · x.
        Uses eigvals for optimal performance.

        The solution is a list of eigenvalues sorted in descending order, where the sorting order
        is defined as: first by the real part (descending), then by the imaginary part (descending).

        :param problem: Tuple (A, B) where A and B are n x n real matrices.
        :return: List of eigenvalues (complex numbers) sorted in descending order.
        """
        A, B = problem
        
        # Convert to numpy arrays with float32 for potentially faster computation
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)

        # Solve generalized eigenvalue problem using eigvals
        eigenvalues = la.eigvals(A, B)

        # Sort eigenvalues: descending order by real part, then by imaginary part.
        # Use numpy argsort for potentially faster sorting
        real_parts = -eigenvalues.real
        imag_parts = -eigenvalues.imag
        # First sort by imaginary part (secondary key), then by real part (primary key)
        indices = np.lexsort((imag_parts, real_parts))
        solution = eigenvalues[indices].tolist()
        return solution