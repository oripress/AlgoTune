from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh_tridiagonal

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n = mat.shape[0]

        if k <= 0:
            return []

        # Fast path for strictly diagonal CSR matrices
        if self._is_diagonal_csr(mat):
            diag = np.real(mat.diagonal())
            if k < n:
                part = np.partition(diag, k - 1)[:k]
                part.sort()
                return [float(x) for x in part]
            diag.sort()
            return [float(x) for x in diag[:k]]

        # Dense path for tiny systems, or k too close to n, or small n
        if k >= n or n < 2 * k + 1 or n <= 64:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Fast path for symmetric tridiagonal matrices (guarded to avoid overhead)
        indptr = mat.indptr
        nnz = mat.nnz
        # A (strict) tridiagonal has at most 3n-2 nonzeros (without duplicates).
        # Also every row must have ≤ 3 nonzeros.
        if nnz <= max(0, 3 * n - 2):
            rows_nnz = indptr[1:] - indptr[:-1]
            if rows_nnz.size and rows_nnz.max(initial=0) <= 3:
                te = self._extract_tridiagonal(mat)
                if te is not None:
                    d, e = te
                    d = np.ascontiguousarray(np.real(d), dtype=float)
                    e = np.ascontiguousarray(np.real(e), dtype=float)
                    if k < n:
                        vals = eigvalsh_tridiagonal(
                            d, e, select="i", select_range=(0, k - 1), check_finite=False
                        )
                    else:
                        vals = eigvalsh_tridiagonal(d, e, check_finite=False)[:k]
                    return [float(v) for v in vals]

        # Sparse Lanczos without shift-invert
        try:
            # For PSD, 'SA' (smallest algebraic) equals 'SM' (smallest magnitude), but often converges faster
            ncv = min(n - 1, max(2 * k + 2, k + 5, 12))
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=ncv,
                tol=1e-8,
            )
        except Exception:
            # Last-resort dense fallback (rare)
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in np.sort(np.real(vals))]

    @staticmethod
    def _is_diagonal_csr(mat: sparse.csr_matrix) -> bool:
        n, m = mat.shape
        if n != m:
            return False
        # Quick reject: more nnz than n cannot be strictly diagonal (ignoring duplicates)
        if mat.nnz > n:
            return False
        indptr = mat.indptr
        indices = mat.indices
        if n == 0:
            return True
        rows_nnz = indptr[1:] - indptr[:-1]
        if rows_nnz.size and rows_nnz.max(initial=0) > 1:
            return False
        mask_one = rows_nnz == 1
        if np.any(mask_one):
            starts = indptr[:-1][mask_one]
            cols = indices[starts]
            rows_idx = np.nonzero(mask_one)[0]
            if np.any(cols != rows_idx):
                return False
        return True

    @staticmethod
    def _extract_tridiagonal(mat: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray] | None:
        # Returns (d, e) if matrix is (symmetric) tridiagonal; otherwise None.
        n, m = mat.shape
        if n != m or n == 0:
            return None
        if n == 1:
            d = mat.diagonal().astype(mat.dtype, copy=False)
            return (np.asarray(d), np.zeros(0, dtype=d.dtype))

        indptr = mat.indptr
        indices = mat.indices
        data = mat.data

        # Quick structural guards
        rows_nnz = indptr[1:] - indptr[:-1]
        if rows_nnz.size and rows_nnz.max(initial=0) > 3:
            return None
        if mat.nnz > max(0, 3 * n - 2):
            return None

        d = np.zeros(n, dtype=data.dtype)
        e_upper = np.zeros(n - 1, dtype=data.dtype)
        e_lower = np.zeros(n - 1, dtype=data.dtype)
        has_upper = np.zeros(n - 1, dtype=bool)
        has_lower = np.zeros(n - 1, dtype=bool)

        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            row_idx = indices[start:end]
            if row_idx.size:
                if row_idx.min() < i - 1 or row_idx.max() > i + 1:
                    return None
            row_dat = data[start:end]
            for col, val in zip(row_idx, row_dat):
                if col == i:
                    d[i] += val
                elif col == i + 1:
                    e_upper[i] += val
                    has_upper[i] = True
                elif col == i - 1:
                    e_lower[i - 1] += val
                    has_lower[i - 1] = True
                else:
                    return None

        e = np.empty(n - 1, dtype=data.dtype)
        tol_base = 1e-12
        for i in range(n - 1):
            if has_upper[i] and has_lower[i]:
                up = e_upper[i]
                lo = e_lower[i]
                scale = max(abs(up), abs(lo), 1.0)
                if abs(up - lo) > tol_base * scale:
                    return None
                e[i] = 0.5 * (up + lo)
            elif has_upper[i]:
                e[i] = e_upper[i]
            elif has_lower[i]:
                e[i] = e_lower[i]
            else:
                e[i] = 0.0

        return d, e