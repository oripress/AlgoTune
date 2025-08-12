import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> np.ndarray:
        """
        Fast PCA using eigen‑decomposition of the smaller Gram matrix.
        - If n_features <= n_samples, compute eigen‑decomposition of XᵀX.
        - Otherwise compute eigen‑decomposition of X Xᵀ and recover V.
        Returns an orthonormal matrix V of shape (n_components, n_features).
        """
        # Load data as float32 for speed
        X = np.asarray(problem["X"], dtype=np.float32)

        # Center columns (features) in‑place
        X -= X.mean(axis=0, keepdims=True)

        n_components = int(problem["n_components"])
        m, n = X.shape

        n_components = int(problem["n_components"])
        m, n = X.shape

        # Fast path when all components are requested
        if n_components >= min(m, n):
            # Full thin SVD is cheap enough in this case
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            V = Vt[:n_components]
        elif n <= m:
            # Compute eigen‑decomposition of the smaller matrix XᵀX (n×n)
            C = X.T @ X
            eigvals, eigvecs = np.linalg.eigh(C)

            # Select top‑k eigenvalues without full sort
            top_idx = np.argpartition(eigvals, -n_components)[-n_components:]
            # Sort the selected indices in descending order
            top_idx = top_idx[np.argsort(eigvals[top_idx])[::-1]]

            V = eigvecs[:, top_idx].T  # shape (n_components, n)
        else:
            # Compute eigen‑decomposition of the smaller matrix X Xᵀ (m×m)
            C = X @ X.T
            eigvals, eigvecs = np.linalg.eigh(C)

            top_idx = np.argpartition(eigvals, -n_components)[-n_components:]
            top_idx = top_idx[np.argsort(eigvals[top_idx])[::-1]]

            eigvecs = eigvecs[:, top_idx]          # shape (m, n_components)
            sigma = np.sqrt(eigvals[top_idx])      # singular values
            nonzero = sigma > 0
            V = np.empty((n_components, n), dtype=X.dtype)
            V[nonzero] = (eigvecs[:, nonzero].T @ X) / sigma[nonzero][:, None]
            V[~nonzero] = 0.0
        # Pad if needed (rank deficiency)
        if V.shape[0] < n_components:
            pad = np.zeros((n_components - V.shape[0], n), dtype=X.dtype)
            V = np.vstack([V, pad])

        return V