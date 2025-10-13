from __future__ import annotations
from typing import Any, List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        """
        Return the k smallest eigenvalues of a (sparse) symmetric positive semi-definite matrix.

        Strategy:
        - Dense path for small problems or when k is close to n (fast and exact enough).
        - Sparse ARPACK (eigsh) otherwise, using 'SA' (smallest algebraic), which equals 'SM' for PSD.
        - Robust sorting and conversion to real floats for safety.
        """
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n: int = mat.shape[0]

        if k <= 0:
            return []

        # Dense path for tiny systems or when k is not much smaller than n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Sparse path: PSD implies smallest algebraic == smallest magnitude
        # Use reasonable subspace size; ensure k < ncv < n.
        ncv = min(n - 1, max(2 * k + 1, 20))
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=ncv,
                tol=0.0,  # strict tolerance to closely match reference
            )
        except Exception:
            # Fallback to dense if ARPACK fails
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Sort and ensure real floats
        vals_sorted = np.sort(np.real(vals))
        return [float(v) for v in vals_sorted[:k]]