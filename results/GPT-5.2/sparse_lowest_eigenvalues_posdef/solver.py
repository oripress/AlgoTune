from __future__ import annotations

from typing import Any, List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        mat = problem["matrix"]
        k = int(problem.get("k", 5))

        if k <= 0:
            return []

        if not sparse.isspmatrix(mat):
            mat = sparse.csr_matrix(mat)
        elif not sparse.isspmatrix_csr(mat):
            asformat = getattr(mat, "asformat", None)
            mat = asformat("csr") if asformat is not None else mat.tocsr()

        n = int(mat.shape[0])
        if n == 0:
            return [0.0] * k

        # Match reference dense behavior for k >= n
        if k >= n:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in np.real(vals[:k])]

        # Fast diagonal CSR path: exactly one entry per row and on the diagonal
        indptr = mat.indptr
        counts = indptr[1:] - indptr[:-1]
        if mat.nnz == n and np.all(counts == 1):
            idx = mat.indices[indptr[:-1]]
            if np.array_equal(idx, np.arange(n, dtype=idx.dtype)):
                d = np.real(mat.data[indptr[:-1]])
                d.sort()
                return [float(v) for v in d[:k]]

        # Dense path (slightly more aggressive than reference for speed)
        nnz = int(mat.nnz)
        if n < 2 * k + 1 or n <= 256 or (n <= 512 and nnz > 0.15 * n * n):
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in np.real(vals[:k])]

        # Sparse ARPACK path.
        # For PSD matrices with tiny eigenvalues, ARPACK without shift-invert can take many iterations.
        # Heuristic: for larger and sufficiently sparse problems, use shift-invert around a small
        # negative sigma (avoids singular factorization at sigma=0 for semidefinite matrices).
        ncv = min(n - 1, max(2 * k + 1, 20))
        nnz_per_row = nnz / n

        if n >= 1200 and nnz_per_row <= 40:
            try:
                d = mat.diagonal()
                scale = float(np.max(np.abs(d))) if d.size else 1.0
                if not np.isfinite(scale) or scale == 0.0:
                    scale = 1.0
                sigma = -1e-3 * scale

                vals = eigsh(
                    mat,
                    k=k,
                    which="LM",
                    sigma=sigma,
                    return_eigenvectors=False,
                    tol=1e-8,
                    maxiter=max(200, n * 5),
                    ncv=ncv,
                )
                vals = np.sort(np.real(vals))[:k]
                return [float(v) for v in vals]
            except Exception:
                pass

        v0 = np.ones(n, dtype=np.float64)
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                tol=1e-7,
                maxiter=max(300, n * 20),
                ncv=ncv,
                v0=v0,
            )
        except ArpackNoConvergence as e:
            vals = e.eigenvalues
            if vals is None or len(vals) < k:
                vals = eigsh(
                    mat,
                    k=k,
                    which="SM",
                    return_eigenvectors=False,
                    tol=1e-10,
                    maxiter=n * 200,
                    ncv=ncv,
                )
        except Exception:
            try:
                vals = eigsh(
                    mat,
                    k=k,
                    which="SM",
                    return_eigenvectors=False,
                    tol=1e-10,
                    maxiter=n * 200,
                    ncv=ncv,
                )
            except Exception:
                vals = np.linalg.eigvalsh(mat.toarray())[:k]

        vals = np.sort(np.real(vals))[:k]
        return [float(v) for v in vals]