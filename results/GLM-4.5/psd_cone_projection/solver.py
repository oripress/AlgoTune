from typing import Any
import numpy as np
from numba import jit

@jit(nopython=True)
def _psd_projection(A):
    """JIT-compiled core computation for PSD projection"""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs * eigvals @ eigvecs.T

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a given positive semidefinite cone projection problem.

        Args:
            problem: A dictionary with problem parameter:
                - A: symmetric matrix.

        Returns:
            A dictionary containing the problem solution:
                - X: result of projecting A onto PSD cone.

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """

        A = np.array(problem["A"])
        X = _psd_projection(A)
        return {"X": X}