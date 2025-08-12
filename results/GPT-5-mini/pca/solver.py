from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Compute PCA principal components.

        Returns:
            V : np.ndarray of shape (n_components, n_features)
                Rows are orthonormal and correspond to top principal directions.
        """
        try:
            X = np.asarray(problem["X"], dtype=np.float64)
            n_components = int(problem["n_components"])

            # Ensure 2-D array
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.ndim != 2:
                raise ValueError("Input X must be 2-D")

            m, n = X.shape

            # Handle trivial / degenerate cases
            if n == 0:
                return np.zeros((n_components, 0), dtype=np.float64)
            if m == 0:
                V = np.zeros((n_components, n), dtype=np.float64)
                for i in range(min(n_components, n)):
                    V[i, i] = 1.0
                return V

            # Center data (subtract column means)
            mean = X.mean(axis=0)
            X = X - mean  # creates centered copy

            # Choose algorithm:
            # - If number of features is not too large (and <= samples), prefer eigen-decomposition
            #   of X^T X (n x n) which is faster when n is smallish.
            # - Otherwise use compact SVD on X (m x n).
            use_eig = (n <= m) and (n <= 1024)

            if use_eig:
                # Compute symmetric matrix and its eigendecomposition
                C = X.T @ X  # shape (n, n)
                vals, vecs = np.linalg.eigh(C)  # vals ascending
                idx = np.argsort(vals)[::-1]
                k = min(n_components, vecs.shape[1])
                # top eigenvectors as rows
                V_top = vecs[:, idx[:k]].T  # shape (k, n)
                if k == n_components:
                    return V_top
                # pad if requested more components than available (rare / invalid)
                V = np.zeros((n_components, n), dtype=np.float64)
                V[:k, :] = V_top
                for i in range(k, n_components):
                    if i < n:
                        V[i, i] = 1.0
                return V

            else:
                # Compact SVD
                U, S, VT = np.linalg.svd(X, full_matrices=False)
                k = min(n_components, VT.shape[0])
                V_top = VT[:k, :].copy()
                if k == n_components:
                    return V_top
                V = np.zeros((n_components, n), dtype=np.float64)
                V[:k, :] = V_top
                for i in range(k, n_components):
                    if i < n:
                        V[i, i] = 1.0
                return V

        except Exception:
            # Fallback: deterministic orthonormal-ish basis (standard basis rows)
            try:
                X = np.asarray(problem.get("X", []), dtype=np.float64)
                n_components = int(problem.get("n_components", 0))
                n = X.shape[1] if X.ndim == 2 else 0
            except Exception:
                n_components = 0
                n = 0
            V = np.zeros((n_components, n), dtype=np.float64)
            for i in range(min(n_components, n)):
                V[i, i] = 1.0
            return V