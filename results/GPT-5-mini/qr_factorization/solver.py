import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Compute the reduced QR factorization of the input matrix.

        Expects problem to be a dict with key "matrix" containing a list of lists
        (or an array-like) representing A (shape n x (n+1) in this task).
        Returns {"QR": {"Q": Q_list, "R": R_list}} where Q and R are lists of lists.
        """
        # Validate input
        if not isinstance(problem, dict) or "matrix" not in problem:
            raise ValueError("Input must be a dict with key 'matrix'.")

        # Convert input to numpy array of floats (avoids unexpected dtypes)
        A = np.asarray(problem["matrix"], dtype=np.float64)

        if A.ndim != 2:
            raise ValueError("'matrix' must be 2-dimensional.")

        # Use NumPy's LAPACK-backed QR factorization (reduced form).
        # For an n x (n+1) matrix this yields Q of shape (n, n) and R of shape (n, n+1).
        Q, R = np.linalg.qr(A, mode="reduced")

        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}