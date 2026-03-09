import math

import numpy as np
from scipy.linalg import svd
from scipy.linalg.blas import dgemm, dsyrk
from scipy.linalg.lapack import dgesdd, dsyevd

class Solver:
    def __init__(self):
        self._array = np.array
        self._asarray = np.asarray
        self._eye = np.eye
        self._hypot = math.hypot
        self._sqrt = np.sqrt
        self._isfinite = np.isfinite
        self._np_eigh = np.linalg.eigh
        self._np_svd = np.linalg.svd
        self._dgemm = dgemm
        self._dsyrk = dsyrk
        self._dgesdd = dgesdd
        self._dsyevd = dsyevd
        self._svd = svd

    def solve(self, problem, **kwargs):
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}

        A = self._asarray(A, dtype=np.float64)
        B = self._asarray(B, dtype=np.float64)

        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
            return {}

        n, m = A.shape
        if n != m:
            return {}

        if n == 1:
            x = float(B[0, 0]) * float(A[0, 0])
            return {"solution": self._array([[1.0 if x >= 0.0 else -1.0]], dtype=np.float64)}

        if n == 2:
            a00 = float(B[0, 0] * A[0, 0] + B[0, 1] * A[0, 1])
            a01 = float(B[0, 0] * A[1, 0] + B[0, 1] * A[1, 1])
            a10 = float(B[1, 0] * A[0, 0] + B[1, 1] * A[0, 1])
            a11 = float(B[1, 0] * A[1, 0] + B[1, 1] * A[1, 1])

            rs = a00 + a11
            rt = a10 - a01
            rden = self._hypot(rs, rt)

            fs = a00 - a11
            ft = a01 + a10
            fden = self._hypot(fs, ft)

            if rden == 0.0 and fden == 0.0:
                return {"solution": self._array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)}

            if abs(rden - fden) <= 1e-12 * max(rden, fden, 1.0):
                M = self._array([[a00, a01], [a10, a11]], dtype=np.float64, order="F")
                U, _, Vt, info = self._dgesdd(M, compute_uv=1, full_matrices=0, overwrite_a=1)
                if info == 0:
                    return {"solution": U @ Vt}
                U, _, Vt = self._svd(
                    M,
                    full_matrices=False,
                    compute_uv=True,
                    overwrite_a=True,
                    check_finite=False,
                )
                return {"solution": U @ Vt}

            if rden > fden:
                c = rs / rden
                s = rt / rden
                return {"solution": self._array([[c, -s], [s, c]], dtype=np.float64)}

            c = fs / fden
            s = ft / fden
            return {"solution": self._array([[c, s], [s, -c]], dtype=np.float64)}

        if n <= 8:
            M = B @ A.T
            C = M.T @ M
            w, V = self._np_eigh(C)
            wmax = float(w[-1])
            if wmax == 0.0:
                return {"solution": self._eye(n, dtype=np.float64)}
            if float(w[0]) > 1e-12 * wmax:
                G = M @ V
                G /= self._sqrt(w)
                return {"solution": G @ V.T}
            U, _, Vt = self._np_svd(M, full_matrices=False)
            return {"solution": U @ Vt}

        M = self._dgemm(alpha=1.0, a=B, b=A, trans_b=True)
        C = self._dsyrk(alpha=1.0, a=M, trans=1, lower=0)
        w, V, info = self._dsyevd(C, compute_v=1, lower=0, overwrite_a=1)
        if info == 0:
            wmax = float(w[-1])
            if wmax == 0.0:
                return {"solution": self._eye(n, dtype=np.float64)}
            if float(w[0]) > 1e-12 * wmax:
                G = self._dgemm(alpha=1.0, a=M, b=V)
                G /= self._sqrt(w)
                G = self._dgemm(alpha=1.0, a=G, b=V, trans_b=True)
                return {"solution": G}

        U, _, Vt, info = self._dgesdd(M, compute_uv=1, full_matrices=0, overwrite_a=1)
        if info != 0:
            U, _, Vt = self._svd(
                M, full_matrices=False, compute_uv=True, overwrite_a=True, check_finite=False
            )

        return {"solution": U @ Vt}