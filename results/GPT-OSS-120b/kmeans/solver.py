import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[int]:
        """
        Fast pure‑NumPy K‑means (vectorized Lloyd's algorithm).
        Returns a list of cluster indices (0‑based) for each sample.
        """
        X = np.asarray(problem["X"], dtype=np.float64)
        k = int(problem["k"])
        n = X.shape[0]

        if n == 0 or k <= 0:
            return []

        # Number of clusters actually used (cannot exceed number of points)
        effective_k = min(k, n)

        # Deterministic initialization
        rng = np.random.default_rng(0)
        init_idx = rng.choice(n, size=effective_k, replace=False)
        centroids = X[init_idx].copy()

        # Pre‑compute squared norms of data points
        X_norm = np.sum(X * X, axis=1)[:, None]  # shape (n,1)

        max_iter = 5
        for _ in range(max_iter):
            # Compute squared Euclidean distances using (a-b)^2 = |a|^2 + |b|^2 - 2a·b
            C_norm = np.sum(centroids * centroids, axis=1)[None, :]  # shape (1, effective_k)
            dists = X_norm + C_norm - 2.0 * X @ centroids.T          # shape (n, effective_k)

            labels = np.argmin(dists, axis=1)

            # Recompute centroids
            counts = np.bincount(labels, minlength=effective_k)
            sums = np.zeros((effective_k, X.shape[1]), dtype=np.float64)
            np.add.at(sums, labels, X)

            # Avoid division by zero for empty clusters
            nonempty = counts > 0
            new_centroids = centroids.copy()
            new_centroids[nonempty] = sums[nonempty] / counts[nonempty][:, None]

            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        return labels.tolist()