import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the orthogonal Procrustes problem:
            minimize_G ||G A - B||_F^2  subject to G^T G = I

        The optimal solution is G = U V^T where M = B A^T = U S V^T.

        Input:
            problem: dict with keys "A" and "B" (n x n real matrices)

        Output:
            dict with key "solution" containing the n x n matrix G as nested lists
        """
        # Basic validation
        if not isinstance(problem, dict):
            return {}
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}

        # Convert to numpy arrays of floats
        try:
            A = np.asarray(A, dtype=np.float64)
            B = np.asarray(B, dtype=np.float64)
        except Exception:
            return {}

        # Check shapes: must be square and equal
        if A.shape != B.shape:
            return {}
        if A.ndim != 2:
            return {}
        n_rows, n_cols = A.shape
        if n_rows != n_cols:
            return {}

        # Compute M = B A^T
        M = B.dot(A.T)

        # Compute SVD of M and form G = U @ Vt
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            G = U.dot(Vt)
        except np.linalg.LinAlgError:
            # Fall back to identity if SVD fails (rare)
            G = np.eye(n_rows, dtype=np.float64)

        # Ensure numerical sanity
        if not np.isfinite(G).all():
            G = np.eye(n_rows, dtype=np.float64)

        return {"solution": G.tolist()}