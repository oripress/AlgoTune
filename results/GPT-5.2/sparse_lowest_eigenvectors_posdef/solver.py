from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat_in = problem["matrix"]
        mat: sparse.csr_matrix = mat_in if sparse.isspmatrix_csr(mat_in) else mat_in.asformat("csr")
        k: int = int(problem["k"])
        n: int = int(mat.shape[0])

        # Fast diagonal-only path.
        # (Common special case; avoids any iterative solver overhead.)
        if mat.nnz == n:
            indptr = mat.indptr
            if np.all((indptr[1:] - indptr[:-1]) == 1):
                idx = mat.indices
                if idx.dtype.kind in ("i", "u") and np.array_equal(idx, np.arange(n, dtype=idx.dtype)):
                    d = np.real(mat.data).astype(np.float64, copy=False)
                    if k < n:
                        out = np.partition(d, k - 1)[:k]
                        out.sort()
                        return [float(v) for v in out]
                    d.sort()
                    return [float(v) for v in d[:k]]

        # Dense path only when it is very likely faster / required.
        if k >= n or n < 2 * k + 1 or n <= 48:
            vals = np.linalg.eigvalsh(mat.toarray())
            vals = np.real(vals[:k]).astype(np.float64, copy=False)
            return [float(v) for v in vals]

        # If matrix is not very sparse and n is moderate, dense partial eigensolve is often faster than ARPACK.
        # dsyevr (via scipy.linalg.eigh subset) computes only the smallest part efficiently.
        if n <= 512:
            # density = nnz / n^2
            if mat.nnz > (n * n) // 6:  # ~ > 0.166 density
                a = mat.toarray()
                vals = eigh(a, eigvals_only=True, subset_by_index=(0, k - 1), driver="evr")
                vals = np.real(vals).astype(np.float64, copy=False)
                return [float(v) for v in vals]

        # Fast sparse path: PSD => smallest algebraic ("SA") are the desired ones.
        # Tune ncv based on sparsity: denser matrices benefit from a larger subspace.
        avg_nnz = mat.nnz // n
        base_ncv = 18 if avg_nnz <= 16 else 32
        ncv = min(n - 1, max(2 * k + 1, base_ncv))

        v0 = np.ones(n, dtype=np.float64)
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                tol=1e-6,
                maxiter=max(160, n * 18),
                ncv=ncv,
                v0=v0,
            )
            vals = np.sort(np.real(vals).astype(np.float64, copy=False))
            return [float(v) for v in vals]
        except ArpackNoConvergence as e:
            ev = getattr(e, "eigenvalues", None)
            if ev is not None and len(ev) >= k:
                vals = np.sort(np.real(np.asarray(ev)))[:k].astype(np.float64, copy=False)
                return [float(v) for v in vals]
        except Exception:
            pass
        # Robust fallback (matches reference behavior more closely).
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
            vals = np.sort(np.real(vals).astype(np.float64, copy=False))
            return [float(v) for v in vals]
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
            vals = np.real(vals).astype(np.float64, copy=False)
            return [float(v) for v in vals]