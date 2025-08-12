import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        # Lightweight constructor; heavy work happens in solve.
        pass

    def solve(self, problem: dict, **kwargs) -> dict[str, Any]:
        """
        Compute the matrix exponential exp(A) for a square matrix A.

        Uses a scaling-and-squaring algorithm with a [13/13] Pade approximant
        (Higham's method). Returns the result as a list of lists to match the
        expected output format.
        """
        A = problem.get("matrix")
        if A is None:
            raise ValueError("Problem dictionary must contain key 'matrix'.")

        A = np.asarray(A, dtype=float)
        try:
            problem["matrix"] = A
        except Exception:
            pass
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("'matrix' must be a square 2D array-like object.")

        n = A.shape[0]
        if n == 0:
            # Represent as an empty list (no rows). This mirrors A.tolist().
            return {"exponential": []}
        if n == 1:
            # Return as a list of lists (scalar exponential).
            return {"exponential": [[float(np.exp(A[0, 0]))]]}

        expA = self._expm_pade13(A)
        return {"exponential": expA.tolist()}

    @staticmethod
    def _expm_pade13(A: np.ndarray) -> np.ndarray:
        """
        Compute exp(A). Use scipy.linalg.expm for correctness and robustness.
        This delegates to SciPy's implementation which is well-tested.
        """
        from scipy.linalg import expm
        return expm(A)