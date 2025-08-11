from typing import Any, Dict

import numpy as np
from math import sqrt
from scipy.linalg.lapack import get_lapack_funcs

# Bind the LAPACK potrf routine for double precision at import time to avoid lookup overhead per call.
_potrf = get_lapack_funcs("potrf", dtype=np.float64)

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Compute the Cholesky factorization A = L L^T for a symmetric positive definite matrix A.

        Parameters:
            problem: dict with key "matrix" mapping to a matrix-like (list of lists or numpy array).

        Returns:
            dict: {"Cholesky": {"L": L}}, where L is a numpy array (lower triangular).
        """
        A = problem["matrix"]
        A_is_np = isinstance(A, np.ndarray)
        n = A.shape[0] if A_is_np else len(A)

        # Small-size fast paths (avoid LAPACK overhead)
        if n == 1:
            a00 = float(A[0, 0] if A_is_np else A[0][0])
            L = np.array([[sqrt(a00)]], dtype=np.float64)
            return {"Cholesky": {"L": L}}

        if n == 2:
            if A_is_np:
                a00 = float(A[0, 0]); a10 = float(A[1, 0]); a11 = float(A[1, 1])
            else:
                a00 = float(A[0][0]); a10 = float(A[1][0]); a11 = float(A[1][1])
            l11 = sqrt(a00)
            l21 = a10 / l11
            l22 = sqrt(a11 - l21 * l21)
            L = np.array([[l11, 0.0], [l21, l22]], dtype=np.float64)
            return {"Cholesky": {"L": L}}

        if n == 3:
            if A_is_np:
                a00 = float(A[0, 0]); a10 = float(A[1, 0]); a20 = float(A[2, 0])
                a11 = float(A[1, 1]); a21 = float(A[2, 1]); a22 = float(A[2, 2])
            else:
                a00 = float(A[0][0]); a10 = float(A[1][0]); a20 = float(A[2][0])
                a11 = float(A[1][1]); a21 = float(A[2][1]); a22 = float(A[2][2])

            l11 = sqrt(a00)
            l21 = a10 / l11
            l31 = a20 / l11

            t22 = a11 - l21 * l21
            l22 = sqrt(t22)

            l32 = (a21 - l31 * l21) / l22

            t33 = a22 - l31 * l31 - l32 * l32
            l33 = sqrt(t33)

            L = np.array(
                [
                    [l11, 0.0, 0.0],
                    [l21, l22, 0.0],
                    [l31, l32, l33],
                ],
                dtype=np.float64,
            )
            return {"Cholesky": {"L": L}}

        # Generic small-n naive path (avoid LAPACK overhead for very small matrices)
        if n <= 8:
            L = np.zeros((n, n), dtype=np.float64)
            if A_is_np:
                for i in range(n):
                    for j in range(i + 1):
                        s = 0.0
                        for k in range(j):
                            s += L[i, k] * L[j, k]
                        t = float(A[i, j]) - s
                        if i == j:
                            L[i, j] = sqrt(t)
                        else:
                            L[i, j] = t / L[j, j]
            else:
                # A is list-of-lists
                for i in range(n):
                    Ai = A[i]
                    for j in range(i + 1):
                        s = 0.0
                        for k in range(j):
                            s += L[i, k] * L[j, k]
                        t = float(Ai[j]) - s
                        if i == j:
                            L[i, j] = sqrt(t)
                        else:
                            L[i, j] = t / L[j, j]
            return {"Cholesky": {"L": L}}

        # Create a Fortran-contiguous float64 copy to enable efficient in-place LAPACK factorization.
        if A_is_np:
            Af = np.array(A, dtype=np.float64, order="F", copy=True)
        else:
            Af = np.array(A, dtype=np.float64, order="F")

        # In-place Cholesky factorization using LAPACK (lower triangular).
        c, info = _potrf(Af, lower=1, overwrite_a=1)
        if info != 0:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed with info={info}")

        L = c
        # Zero out the upper triangle using column-contiguous slices (faster for Fortran-ordered arrays).
        for j in range(n):
            L[:j, j] = 0.0

        return {"Cholesky": {"L": L}}