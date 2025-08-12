import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from typing import Any, Tuple

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray], **kwargs) -> Any:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        
        The problem is defined as: A · x = λ B · x.
        For better numerical stability, we first scale B, then solve the problem.
        
        The solution is a list of eigenvalues sorted in descending order, where the sorting order
        is defined as: first by the real part (descending), then by the imaginary part (descending).
        
        :param problem: Tuple (A, B) where A and B are n x n real matrices.
        :return: List of eigenvalues (complex numbers) sorted in descending order.
        """
        A, B = problem
        
        # Direct computation without scaling for better performance
        eigenvalues = la.eigvals(A, B)
        
        # More efficient sorting using view as real for complex numbers
        # This treats each complex number as a pair of floats (real, imag)
        eigenview = eigenvalues.view(float).reshape(-1, 2)
        
        # Sort by -real, then by -imag using lexsort
        indices = np.lexsort((-eigenview[:, 1], -eigenview[:, 0]))
        
        # Return sorted eigenvalues as a list
        return eigenvalues[indices].tolist()