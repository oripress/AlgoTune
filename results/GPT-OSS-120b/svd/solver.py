import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the singular value decomposition of the given matrix.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "matrix": 2D list or array representing matrix A.

        Returns
        -------
        dict
            Dictionary with keys:
                "U": numpy.ndarray of shape (n, k)
                "S": numpy.ndarray of shape (k,)
                "V": numpy.ndarray of shape (m, k)
        """
        # Perform SVD in double‑precision (float64) for numerical accuracy.
        A = np.asarray(problem["matrix"], dtype=np.float64)
        n, m = A.shape
        if n >= m:
            U, s, Vh = np.linalg.svd(A, full_matrices=False)
            V = Vh.T
        else:
            # Compute SVD of the transpose to work on the smaller dimension
            Ut, s, Vht = np.linalg.svd(A.T, full_matrices=False)
            V = Ut                     # shape (m, k)
            U = Vht.T                  # shape (n, k)
        return {"U": U, "S": s, "V": V}