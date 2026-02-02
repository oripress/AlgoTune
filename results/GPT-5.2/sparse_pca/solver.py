from __future__ import annotations

from typing import Any

import numpy as np

try:
    # Dense LAPACK interface; supports fast subset eigenpairs.
    from scipy.linalg import eigh as _sp_eigh  # type: ignore
except Exception:  # pragma: no cover
    _sp_eigh = None

class Solver:
    __slots__ = ()

    def solve(self, problem: dict, **kwargs: Any) -> dict:
        """
        Fast closed-form solution of the reference convex Sparse PCA surrogate.

        Per column b, solve:
            min_x ||b - x||_2^2 + lam*||x||_1  s.t. ||x||_2 <= 1
        => x = proj_{||.||2<=1}(soft_threshold(b, lam/2))
        """
        A = np.asarray(problem["covariance"], dtype=np.float64)
        n_components = int(problem["n_components"])
        lam = float(problem["sparsity_param"])

        n = int(A.shape[0])
        if n_components <= 0:
            return {
                "components": np.empty((n, 0), dtype=np.float64),
                "explained_variance": np.empty((0,), dtype=np.float64),
            }

        k_req = n_components if n_components < n else n

        # For small matrices, NumPy's eigh is typically lower overhead.
        # For larger matrices with small k, SciPy's subset can be much faster.
        use_scipy_subset = (
            _sp_eigh is not None and 0 < k_req < n and n >= 80 and k_req <= (n // 3)
        )

        if use_scipy_subset:
            Af = np.array(A, dtype=np.float64, order="F", copy=False)
            lo = n - k_req
            eigvals, eigvecs = _sp_eigh(
                Af,
                subset_by_index=(lo, n - 1),
                overwrite_a=True,
                check_finite=False,
                driver="evr",
            )
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
        else:
            eigvals, eigvecs = np.linalg.eigh(A)  # ascending
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]

        # Reference keeps strictly positive eigenvalues.
        if eigvals[-1] <= 0.0:
            eigvals_asc = eigvals[::-1]
            i0 = int(np.searchsorted(eigvals_asc, 0.0, side="right"))
            if i0 >= eigvals_asc.shape[0]:
                X = np.zeros((n, n_components), dtype=np.float64)
                return {
                    "components": X,
                    "explained_variance": np.zeros((n_components,), dtype=np.float64),
                }
            eigvals = eigvals_asc[i0:][::-1]
            eigvecs = eigvecs[:, : eigvals.shape[0]]
        else:
            if eigvals.shape[0] > k_req:
                eigvals = eigvals[:k_req]
                eigvecs = eigvecs[:, :k_req]

        kpos = int(eigvals.shape[0])
        k_use = n_components if kpos >= n_components else kpos

        # B = eigvecs[:, :k] * sqrt(eigvals[:k]) (columnwise scaling).
        Xk = eigvecs[:, :k_use] * np.sqrt(eigvals[:k_use])

        t = 0.5 * lam
        if t != 0.0:
            sign = np.sign(Xk)
            np.abs(Xk, out=Xk)
            Xk -= t
            np.maximum(Xk, 0.0, out=Xk)
            Xk *= sign

        norms = np.linalg.norm(Xk, axis=0)
        np.maximum(norms, 1.0, out=norms)
        Xk /= norms

        # Explained variance: only compute for nonzero columns (k_use).
        AXk = A @ Xk
        explained_k = np.sum(Xk * AXk, axis=0)

        if k_use == n_components:
            X = Xk
            explained = explained_k
        else:
            X = np.zeros((n, n_components), dtype=np.float64)
            X[:, :k_use] = Xk
            explained = np.zeros((n_components,), dtype=np.float64)
            explained[:k_use] = explained_k

        return {"components": X, "explained_variance": explained}