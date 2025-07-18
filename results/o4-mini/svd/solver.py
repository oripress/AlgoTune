import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

class Solver:
    def __init__(self):
        # Prebind LAPACK divide-and-conquer SVD for fallback
        dummy = np.empty((1, 1), dtype=np.float64, order='F')
        (self._gesdd,) = get_lapack_funcs(('gesdd',), (dummy,))

    def solve(self, problem, **kwargs):
        """
        Compute the singular value decomposition of matrix A.
        Uses Gram-based eigen-decomposition for speed, with SVD fallback for
        ill-conditioned or rank-deficient cases.
        """
        # Load data as Fortran-ordered float64 array for LAPACK
        A = np.array(problem["matrix"], dtype=np.float64, order='F', copy=True)
        n, m = A.shape
        k = min(n, m)
        # Tolerance for detecting rank deficiency
        eps = np.finfo(np.float64).eps
        tol = max(n, m) * eps
        if n >= m:
            # Eigen-decomposition on A^T A (size m x m)
            B = A.T.dot(A)
            w, V = np.linalg.eigh(B)
            idx = np.argsort(w)[::-1]
            w = w[idx]
            V = V[:, idx]
            w[w < 0] = 0.0
            s = np.sqrt(w)
            # Fallback if singular values too small
            if np.any(s <= tol):
                u, s, vt, _ = self._gesdd(A, compute_uv=1,
                                          full_matrices=0,
                                          overwrite_a=1)
                return {"U": u, "S": s, "V": vt.T}
            U = A.dot(V) / s
            return {"U": U, "S": s, "V": V}
        else:
            # Eigen-decomposition on A A^T (size n x n)
            C = A.dot(A.T)
            w, U = np.linalg.eigh(C)
            idx = np.argsort(w)[::-1]
            w = w[idx]
            U = U[:, idx]
            w[w < 0] = 0.0
            s = np.sqrt(w)
            if np.any(s <= tol):
                u, s, vt, _ = self._gesdd(A, compute_uv=1,
                                          full_matrices=0,
                                          overwrite_a=1)
                return {"U": u, "S": s, "V": vt.T}
            V = A.T.dot(U) / s
            return {"U": U, "S": s, "V": V}