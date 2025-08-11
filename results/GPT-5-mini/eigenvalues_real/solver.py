"""
Fast solver for the EigenvaluesReal task.

Uses numpy.linalg.eigvalsh which computes eigenvalues only for symmetric/Hermitian
matrices (faster than computing eigenvectors as well). The solver converts the
input to a numpy array, retries with a symmetrized matrix if necessary, sorts
the eigenvalues in descending order, and returns a Python list of floats.
"""

from typing import Any, List
import numpy as np

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[float]:
        """
        Compute eigenvalues of a real symmetric matrix.

        :param problem: n x n matrix (list of lists or numpy array)
        :return: list of eigenvalues sorted in descending order
        """
        # Convert input to numpy array of floats (no copy when possible)
        A = np.array(problem, dtype=float, copy=False)

        # Basic validation: must be square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square (n x n) matrix.")

        # Use eigvalsh which is optimized for symmetric/Hermitian matrices and
        # returns eigenvalues only (faster and less memory than eigh).
        try:
            vals = np.linalg.eigvalsh(A)
        except Exception:
            # If numerical asymmetry causes an issue, enforce symmetry and retry.
            A = 0.5 * (A + A.T)
            vals = np.linalg.eigvalsh(A)

        # Sort in descending order and return as a Python list of floats.
        vals.sort()
        return vals[::-1].tolist()