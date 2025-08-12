from typing import Any, Dict

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, np.ndarray]:
        """
        Compute the singular value decomposition (SVD) of matrix A.

        Returns U, S, V such that A ≈ U @ diag(S) @ V.T with shapes:
          - U: (n, k)
          - S: (k,)
          - V: (m, k)
        where k = min(n, m).
        """
        A = problem["matrix"]
        mat = np.asarray(A, dtype=float)
        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
        V = Vh.T
        return {"U": U, "S": s, "V": V}