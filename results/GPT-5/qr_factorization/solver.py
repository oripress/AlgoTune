from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    from scipy.linalg import qr as scipy_qr  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        """
        Compute QR of A (n x (n+1)) via:
          - QR on the first n columns (square block)
          - Project last column with Q^T
        """
        A = problem["matrix"]
        A_arr = A if isinstance(A, np.ndarray) and A.dtype == np.float64 else np.asarray(A, dtype=np.float64)

        n = A_arr.shape[0]
        # Heuristic: SciPy QR can be faster for larger n due to efficient LAPACK bindings
        if _HAS_SCIPY and n >= 96:
            Q, R0 = scipy_qr(A_arr[:, :n], mode="economic", check_finite=False, overwrite_a=False)  # type: ignore
        else:
            # Use Fortran-ordered copy of the square block for faster LAPACK QR in NumPy
            Bf = np.asfortranarray(A_arr[:, :n])
            Q, R0 = np.linalg.qr(Bf, mode="reduced")

        # Assemble R = [R0 | Q^T c] directly into the last column to avoid a temp
        R = np.empty((n, n + 1), dtype=A_arr.dtype, order="F")
        R[:, :n] = R0
        np.matmul(Q.T, A_arr[:, n], out=R[:, n])

        return {"QR": {"Q": Q, "R": R}}