from typing import Any, Dict

import numpy as np

try:
    from scipy.linalg import qr as sp_qr
    _HAVE_SCIPY = True
except Exception:
    sp_qr = None  # type: ignore
    _HAVE_SCIPY = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute the QR factorization of the input matrix A using a fast LAPACK-backed routine.
        Prefers SciPy's QR (with overwrite and no finiteness checks) for performance; falls back to NumPy.
        """
        # Ensure float64 and Fortran memory layout to accelerate LAPACK.
        A = np.array(problem["matrix"], dtype=np.float64, order="F", copy=False)

        # Handle empty edge-case quickly
        m, n = A.shape
        if m == 0:
            return {"QR": {"Q": np.empty((0, 0), dtype=np.float64), "R": np.empty((0, n), dtype=np.float64)}}

        if _HAVE_SCIPY:
            Q, R = sp_qr(A, mode="economic", overwrite_a=True, check_finite=False)
        else:
            Q, R = np.linalg.qr(A, mode="reduced")

        return {"QR": {"Q": Q, "R": R}}