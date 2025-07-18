from typing import Any
import numpy as np

# Try to import randomized SVD from scikit-learn
try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        # Load data and parameters
        X = np.array(problem.get("X", []), dtype=float)
        n_components = int(problem.get("n_components", 0))
        # Validate input
        if X.ndim != 2 or X.size == 0 or n_components <= 0:
            return []
        # Center data (in-place)
        X -= X.mean(axis=0)
        m, n = X.shape
        # Effective number of components
        k = min(n_components, m, n)
        if k <= 0:
            return []
        # Decide on algorithm: randomized SVD if available and k is small
        min_dim = min(m, n)
        # Decide on algorithm: randomized SVD if available and k is small relative to dims
        min_dim = min(m, n)
        if randomized_svd is not None and k * 4 < min_dim:
            # randomized SVD with small oversampling and 1 power iteration for speed
            U, S, VT = randomized_svd(
                X, n_components=k, n_oversamples=10, n_iter=1, random_state=None
            )
            return VT.tolist()
        if n <= m:
            # covariance in feature space: (n x n)
            cov = X.T @ X
            vals, vecs = np.linalg.eigh(cov)
            # pick top k eigenvectors
            idx = np.argsort(vals)[::-1][:k]
            V = vecs[:, idx].T  # shape (k, n)
        else:
            # covariance in sample space: (m x m)
            cov = X @ X.T
            vals, vecs = np.linalg.eigh(cov)
            idx = np.argsort(vals)[::-1][:k]
            U = vecs[:, idx]    # (m, k)
            Vmat = X.T @ U      # (n, k)
            V = (Vmat / np.sqrt(vals[idx])[None, :]).T  # (k, n)
        return V.tolist()