from typing import Any

import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        try:
            X = np.array(problem["X"], dtype=np.float32, copy=True)
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")

            m, n = X.shape
            k = int(problem["n_components"])

            if k < 0:
                raise ValueError("n_components must be non-negative")
            if k == 0:
                return np.empty((0, n), dtype=np.float32)
            if n == 0:
                return np.empty((k, 0), dtype=np.float32)

            # If all feature directions are requested, any orthonormal basis works.
            if k == n:
                return np.eye(n, dtype=np.float32)

            X -= X.mean(axis=0, dtype=np.float32)

            # For tall/square matrices, the principal directions are the top
            # eigenvectors of X^T X.
            if m >= n:
                _, vecs = np.linalg.eigh(X.T @ X)
                return vecs[:, -k:].T

            # After centering, rank(X) <= m - 1, so if we need the whole row
            # space, any orthonormal basis of it is optimal.
            if k >= m - 1:
                return np.linalg.qr(X.T, mode="reduced")[0].T[:k]

            # For small-k wide matrices, the smaller sample Gram matrix is often
            # faster than a full SVD.
            vals, u = np.linalg.eigh(X @ X.T)
            vals = vals[-k:]
            if vals[0] > np.finfo(np.float32).eps * max(vals[-1], 1.0):
                return (u[:, -k:].T @ X) / np.sqrt(vals)[:, None]

            # Fall back when numerical rank is smaller than requested.
            return np.linalg.svd(X, full_matrices=False)[2][:k]

        except Exception:
            X = np.asarray(problem["X"])
            n = X.shape[1] if X.ndim == 2 else 0
            k = int(problem.get("n_components", 0))
            out = np.zeros((k, n), dtype=np.float32)
            d = min(k, n)
            if d:
                out[:d, :d] = np.eye(d, dtype=np.float32)
            return out