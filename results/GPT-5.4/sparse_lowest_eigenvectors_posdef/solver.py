from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg as sla
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    @staticmethod
    def _dense_smallest(mat: sparse.spmatrix, k: int) -> list[float]:
        arr = mat.toarray(order="F")
        n = arr.shape[0]
        if k >= n:
            vals = sla.eigh(
                arr,
                eigvals_only=True,
                overwrite_a=True,
                check_finite=False,
                driver="evr",
            )
            return [float(v) for v in np.real(vals[:k])]
        vals = sla.eigh(
            arr,
            eigvals_only=True,
            subset_by_index=(0, k - 1),
            overwrite_a=True,
            check_finite=False,
            driver="evr",
        )
        return [float(v) for v in np.real(vals)]

    def solve(self, problem, **kwargs) -> Any:
        mat = problem["matrix"]
        if not sparse.isspmatrix_csr(mat):
            mat = mat.tocsr(copy=False)

        k = int(problem["k"])
        n = mat.shape[0]

        if k <= 0:
            return []

        if k >= n or n < 2 * k + 1:
            return self._dense_smallest(mat, k)

        nnz = int(mat.nnz)
        if nnz == 0:
            return [0.0] * k

        # Exact shortcut for purely diagonal CSR matrices.
        if nnz == n:
            indptr = mat.indptr
            if np.all(indptr[1:] - indptr[:-1] == 1):
                idx = mat.indices
                if idx.shape[0] == n and np.array_equal(idx, np.arange(n, dtype=idx.dtype)):
                    vals = np.sort(np.real(mat.data))
                    return [float(v) for v in vals[:k]]

        # Dense can be faster for small or moderately dense systems.
        if n <= 160 or nnz * 20 >= n * n:
            return self._dense_smallest(mat, k)

        data = mat.data
        scale = float(np.max(np.abs(data)))
        if scale == 0.0:
            return [0.0] * k

        # For PSD matrices, smallest algebraic equals smallest magnitude, but
        # "SA" is an extremal problem and is often much faster than "SM".
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                tol=0.0,
                ncv=min(n - 1, max(2 * k + 1, 12)),
                maxiter=max(80, n * 8),
            )
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals]
        except Exception:
            pass

        # Secondary fast path: shift-invert close to zero.
        try:
            vals = eigsh(
                mat,
                k=k,
                sigma=-1e-9 * scale,
                which="LM",
                return_eigenvectors=False,
                tol=0.0,
                ncv=min(n - 1, max(2 * k + 1, 12)),
                maxiter=max(60, n * 10),
            )
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals]
        except Exception:
            pass

        # Match reference behavior on fallback.
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            return self._dense_smallest(mat, k)

        vals = np.sort(np.real(vals))
        return [float(v) for v in vals]
        vals = np.sort(np.real(vals))
        return [float(v) for v in vals]