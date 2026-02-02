from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import scipy.linalg

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat: sparse.spmatrix = problem["matrix"]
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Ensure CSR format for efficient arithmetic and slicing
        if not sparse.isspmatrix_csr(mat):
            mat = mat.tocsr()

        # Dense path for smaller systems or when k is large relative to n
        # scipy.linalg.eigh with subset_by_index is very efficient
        if n < 2000 or k >= n:
            try:
                # subset_by_index is (min_index, max_index) inclusive
                # We want the first k eigenvalues (indices 0 to k-1)
                vals = scipy.linalg.eigh(
                    mat.toarray(), 
                    subset_by_index=(0, k-1), 
                    eigvals_only=True,
                    overwrite_a=True,
                    check_finite=False
                )
                return [float(v) for v in vals]
            except Exception:
                pass

        # Sparse Lanczos with shift-invert
        # This is usually the fastest for interior/smallest eigenvalues of sparse matrices
        # We use a small negative sigma to target eigenvalues near 0.
        # Since A is PSD, eigenvalues are >= 0.
        try:
            vals = eigsh(
                mat,
                k=k,
                sigma=-1e-5,
                which="LM",  # Largest Magnitude of (A - sigma I)^-1
                return_eigenvectors=False,
                maxiter=n * 100,
                ncv=min(n - 1, max(2 * k + 1, 20)),
                tol=1e-6
            )
        except Exception:
            # Fallback to standard Lanczos (SA - Smallest Algebraic)
            try:
                vals = eigsh(
                    mat,
                    k=k,
                    which="SA",
                    return_eigenvectors=False,
                    maxiter=n * 200,
                    ncv=min(n - 1, max(2 * k + 1, 20)),
                    tol=1e-6
                )
            except Exception:
                # Last‑resort dense fallback
                vals = scipy.linalg.eigh(
                    mat.toarray(), 
                    subset_by_index=(0, k-1), 
                    eigvals_only=True,
                    overwrite_a=True,
                    check_finite=False
                )

        return [float(v) for v in np.sort(np.real(vals))]