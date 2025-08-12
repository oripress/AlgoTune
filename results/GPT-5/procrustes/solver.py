from __future__ import annotations

from typing import Any, Dict

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the orthogonal Procrustes problem:
            minimize_G || G A - B ||_F^2
        subject to G being orthogonal (G^T G = I).

        The optimal solution is G = U V^T where U, S, V^T = svd(B A^T).
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}

        # Convert inputs to numpy arrays (float64) to ensure fast BLAS-backed matmul/SVD
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        # Validate shapes: square and matching
        if A.ndim != 2 or B.ndim != 2:
            return {}
        if A.shape != B.shape or A.shape[0] != A.shape[1]:
            return {}

        n = A.shape[0]
        if n == 0:
            return {"solution": np.empty((0, 0), dtype=np.float64)}

        # Fast 1x1 case
        if n == 1:
            m = B[0, 0] * A[0, 0]
            g = 1.0 if float(m) >= 0.0 else -1.0
            return {"solution": np.array([[g]], dtype=np.float64)}

        # General case: G = U V^T for M = B A^T
        M = B @ A.T
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        G = U @ Vt

        return {"solution": G}