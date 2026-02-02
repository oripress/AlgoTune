from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.linalg import eigh as _sp_eigh  # type: ignore
except Exception:  # pragma: no cover
    _sp_eigh = None

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        # float32 is typically faster; validator tolerances are 1e-4
        X = np.asarray(problem["X"], dtype=np.float32, order="C")
        k = int(problem["n_components"])

        m, n = X.shape
        if k <= 0:
            return np.zeros((0, n), dtype=np.float32)

        # Trivial cases: centered data has zero variance if there's <=1 sample.
        if m <= 1:
            return np.eye(n, dtype=np.float32)[:k]

        # If requesting full feature basis, any orthonormal basis is optimal.
        if k >= n:
            return np.eye(n, dtype=np.float32)

        # Mean (needed in all non-trivial cases)
        mu = X.mean(axis=0, dtype=np.float32)

        mn = m * n
        # For small/medium problems, a single SVD is usually fastest.
        if (m <= 128 and n <= 128) or (mn <= 60_000):
            if not X.flags.writeable:
                X = X.copy()
            X -= mu
            return np.linalg.svd(X, full_matrices=False)[2][:k]

        # Helpers for partial eigensolves
        def _topk_sym(A: np.ndarray, kk: int) -> tuple[np.ndarray, np.ndarray]:
            """Return (w_desc, V_cols_desc) for symmetric A."""
            dim = A.shape[0]
            if _sp_eigh is not None and dim >= 256 and (kk * 4) < dim:
                w, V = _sp_eigh(
                    A,
                    subset_by_index=(dim - kk, dim - 1),
                    overwrite_a=True,
                    check_finite=False,
                    driver="evr",
                )
                return w[::-1], V[:, ::-1]
            w, V = np.linalg.eigh(A)
            return w[-1 : -kk - 1 : -1], V[:, -1 : -kk - 1 : -1]

        # Tall: work in feature space without explicitly forming centered X
        if m >= n:
            # C = (X - mu)^T (X - mu) = X^T X - m * mu^T mu
            C = X.T @ X
            C -= (np.float32(m) * np.outer(mu, mu))
            _, Vcols = _topk_sym(C, k)
            return Vcols.T

        # Wide: work in sample space without explicitly forming centered X
        # G = (X - mu)(X - mu)^T = XX^T - a[:,None] - a[None,:] + b
        # where a = X @ mu, b = mu·mu
        G = X @ X.T
        a = X @ mu
        G -= a[:, None]
        G -= a[None, :]
        G += np.float32(mu @ mu)

        w, Ucols = _topk_sym(G, k)  # Ucols: (m, k)
        w = np.maximum(w, np.float32(0.0))
        s = np.sqrt(w)

        eps = np.float32(1e-12)
        if s[-1] <= eps:
            # Degenerate: recover and re-orthonormalize
            s = np.maximum(s, eps)
            XtU = X.T @ Ucols
            XtU -= np.outer(mu, Ucols.sum(axis=0, dtype=np.float32))
            V = XtU / s
            Q, _ = np.linalg.qr(V.astype(np.float64), mode="reduced")
            return Q.T.astype(np.float32)

        # V = (Xc^T U) / s, with Xc^T U = X^T U - mu ⊗ (1^T U)
        XtU = X.T @ Ucols
        XtU -= np.outer(mu, Ucols.sum(axis=0, dtype=np.float32))
        V = XtU / s
        return V.T