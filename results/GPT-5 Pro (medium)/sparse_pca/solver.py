from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    # Soft-thresholding operator: S_t(x) = sign(x) * max(|x| - t, 0)
    if t <= 0.0:
        # For t <= 0, standard formula still works but avoid extra allocations
        return np.sign(x) * (np.abs(x) - t)
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def _compute_B(A: np.ndarray, n_components: int) -> np.ndarray:
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    # Keep only positive eigenvalues (as in reference/validator)
    pos_idx = eigvals > 0
    eigvals = eigvals[pos_idx]
    eigvecs = eigvecs[:, pos_idx]
    # Sort in descending order
    if eigvals.size > 0:
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    # Form B with top n_components eigenvectors scaled by sqrt(eigenvalues)
    k = min(len(eigvals), n_components)
    if k == 0:
        # No positive eigenvalues; B is zero
        n = A.shape[0]
        return np.zeros((n, n_components), dtype=float)
    B_top = eigvecs[:, :k] * np.sqrt(eigvals[:k])
    # If fewer than requested components, pad with zeros to maintain shape (n, n_components)
    if k < n_components:
        n = A.shape[0]
        B = np.zeros((n, n_components), dtype=float)
        B[:, :k] = B_top
        return B
    return B_top

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fast solver for sparse PCA surrogate:
        minimize ||B - X||_F^2 + λ ||X||_1 subject to ||X_i||_2 ≤ 1, for each column i.

        Closed-form per-column solution:
        1) x0 = soft_threshold(b, λ/2)
        2) If ||x0||_2 <= 1: x = x0
           else: x = x0 / ||x0||_2  (projection onto unit l2-ball)
        """
        try:
            A = np.array(problem["covariance"], dtype=float)
            n_components = int(problem["n_components"])
            sparsity_param = float(problem["sparsity_param"])
        except Exception:
            return {"components": [], "explained_variance": []}

        # Basic validation
        if A.ndim != 2 or A.shape[0] != A.shape[1] or n_components <= 0 or not np.isfinite(sparsity_param):
            return {"components": [], "explained_variance": []}

        n = A.shape[0]

        # Compute B as in the reference/validator
        B = _compute_B(A, n_components)  # shape (n, n_components)

        # If B has fewer columns (due to padding), it already has shape (n, n_components)
        # Solve per column
        X = np.empty((n, n_components), dtype=float)
        t = sparsity_param / 2.0

        # Vectorized soft-thresholding per column followed by scaling if needed
        for i in range(n_components):
            b = B[:, i]
            x0 = _soft_threshold(b, t)
            norm_x0 = np.linalg.norm(x0)
            if not np.isfinite(norm_x0):
                return {"components": [], "explained_variance": []}
            if norm_x0 <= 1.0 or norm_x0 == 0.0:
                X[:, i] = x0
            else:
                X[:, i] = x0 / norm_x0  # project onto unit l2-ball

        # Compute explained variance for each component
        # Ensure no NaNs or infs
        if not np.all(np.isfinite(X)):
            return {"components": [], "explained_variance": []}

        explained_variance: List[float] = []
        # Efficient multiplication: use matrix-vector products
        for i in range(n_components):
            xi = X[:, i]
            Ax = A @ xi
            var = float(xi.dot(Ax))
            # Guard against tiny negative due to numerical error
            if not np.isfinite(var):
                var = 0.0
            explained_variance.append(var)

        return {
            "components": X.tolist(),
            "explained_variance": explained_variance,
        }