import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.cluster import KMeans
class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Perform spectral clustering using the normalized affinity and k-means++.
        """
        # Extract inputs
        W = problem.get("similarity_matrix")
        k = problem.get("n_clusters")
        if W is None or k is None:
            raise ValueError("Missing similarity_matrix or n_clusters.")
        W = np.asarray(W, dtype=float)
        n, m = W.shape
        if n != m:
            raise ValueError("Similarity matrix must be square.")
        # Edge cases
        if n == 0:
            return {"labels": np.array([], dtype=int), "n_clusters": k}
        if k >= n:
            return {"labels": np.arange(n, dtype=int), "n_clusters": k}
        # Normalize affinity: M = D^{-1/2} W D^{-1/2}
        d = W.sum(axis=1)
        d_inv_sqrt = np.zeros_like(d)
        nz = d > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        M = (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
        # Compute top-k eigenvectors
        try:
            _, vecs = eigsh(M, k=k, which='LA')
        except Exception:
            _, vecs_full = np.linalg.eigh(M)
            vecs = vecs_full[:, -k:]
        # Row-normalize eigenvectors
        U = vecs
        norms = np.linalg.norm(U, axis=1)
        norms[norms == 0] = 1
        U_norm = U / norms[:, None]
        # k-means++ init
        rng = np.random.default_rng()
        centers = np.empty((k, U_norm.shape[1]), dtype=float)
        centers[0] = U_norm[rng.integers(0, n)]
        closest = np.sum((U_norm - centers[0])**2, axis=1)
        for ci in range(1, k):
            probs = closest / closest.sum()
            idx = rng.choice(n, p=probs)
            centers[ci] = U_norm[idx]
            dist_sq = np.sum((U_norm - centers[ci])**2, axis=1)
            closest = np.minimum(closest, dist_sq)
        # Lloyd's algorithm
        labels = np.zeros(n, dtype=int)
        for _ in range(100):
            dists = np.sum((U_norm[:, None, :] - centers[None, :, :])**2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(k):
                pts = U_norm[labels == ci]
                if pts.size:
                    centers[ci] = pts.mean(axis=0)
        return {"labels": labels, "n_clusters": k}