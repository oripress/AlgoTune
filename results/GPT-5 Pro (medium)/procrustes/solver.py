from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    # SciPy SVD can be faster with check_finite=False and overwrite_a=True
    from scipy.linalg import svd as scipy_svd
except Exception:  # pragma: no cover
    scipy_svd = None

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        """
        Solve the orthogonal Procrustes problem:
            minimize_G ||GA - B||_F^2  subject to G^T G = I

        The optimal solution is G = U V^T where U, S, V^T = svd(B A^T).

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" and "B" representing n x n real matrices.

        Returns
        -------
        dict
            Dictionary with key "solution" containing the optimal orthogonal matrix G.
            Returns {} if inputs are invalid.
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}

        # Ensure float64 arrays; use Fortran order to better align with LAPACK expectations
        A = np.array(A, dtype=np.float64, order="F", copy=False)
        B = np.array(B, dtype=np.float64, order="F", copy=False)

        if A.shape != B.shape:
            return {}

        n, m = A.shape
        if n != m:
            return {}

        # Fast path for 1x1: G = sign(B*A) with sign(0) -> +1 to match typical SVD output
        if n == 1:
            val = float(B[0, 0] * A[0, 0])
            g11 = 1.0 if val >= 0.0 else -1.0
            return {"solution": [[g11]]}

        # Compute M = B A^T
        M = B @ A.T

        # Compute SVD of M
        if scipy_svd is not None:
            U, _, Vt = scipy_svd(
                M, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver="gesdd"
            )
        else:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Compute G = U V^T
        G = U @ Vt

        return {"solution": G.tolist()}