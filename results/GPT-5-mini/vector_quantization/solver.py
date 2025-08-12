from typing import Any, Tuple
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Vector quantization (k-means). Preference order:
        1) faiss (if available) - matches reference
        2) scikit-learn (if available)
        3) NumPy fallback: k-means++ init + Lloyd (with empty-cluster fixes)

        Returns a dict with keys: "centroids", "assignments", "quantization_error".
        """
        vectors = problem.get("vectors")
        if vectors is None:
            raise ValueError("Problem must contain 'vectors' key.")

        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Vectors must be a 2D array-like structure.")
        n, dim = X.shape

        k = int(problem.get("k", 0))
        if k <= 0:
            raise ValueError("k must be > 0")

        # Trivial case
        if k == 1:
            centroid = np.mean(X, axis=0)
            diffs = X - centroid
            q_err = float(np.mean(np.sum(diffs * diffs, axis=1)))
            return {
                "centroids": [centroid.tolist()],
                "assignments": [0] * n,
                "quantization_error": q_err,
            }

        # Try faiss (preferred, like the reference)
        try:
            import faiss  # type: ignore

            xb = X.astype(np.float32)
            kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
            kmeans.train(xb)
            centroids = kmeans.centroids.astype(np.float32)
            # Assign points to centroids
            index = faiss.IndexFlatL2(dim)
            index.add(centroids)
            distances, assignments = index.search(xb, 1)
            assignments = assignments.ravel().astype(int)
            quantization_error = float(np.mean(distances))  # distances are squared L2
            sizes = np.bincount(assignments, minlength=k)
            if np.any(sizes == 0):
                centroids, assignments = _fix_empty_clusters(xb, centroids, assignments)
                diffs = xb - centroids[assignments]
                quantization_error = float(np.mean(np.sum(diffs * diffs, axis=1)))
            return {
                "centroids": centroids.tolist(),
                "assignments": assignments.tolist(),
                "quantization_error": quantization_error,
            }
        except Exception:
            pass

        # Try scikit-learn
        try:
            from sklearn.cluster import KMeans, MiniBatchKMeans  # type: ignore

            if n > 50000:
                mb = MiniBatchKMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=3,
                    max_iter=200,
                    batch_size=1024,
                    random_state=0,
                )
                mb.fit(X)
                centroids = mb.cluster_centers_.astype(np.float32)
                assignments = mb.labels_.astype(int)
            else:
                km = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=4,
                    max_iter=200,
                    random_state=0,
                )
                km.fit(X)
                centroids = km.cluster_centers_.astype(np.float32)
                assignments = km.labels_.astype(int)

            sizes = np.bincount(assignments, minlength=k)
            if np.any(sizes == 0):
                centroids, assignments = _fix_empty_clusters(X, centroids, assignments)

            diffs = X - centroids[assignments]
            quantization_error = float(np.mean(np.sum(diffs * diffs, axis=1)))
            return {
                "centroids": centroids.tolist(),
                "assignments": assignments.tolist(),
                "quantization_error": quantization_error,
            }
        except Exception:
            pass

        # NumPy fallback
        centroids, assignments = _kmeans_numpy(X, k, max_iter=200, random_state=0)
        diffs = X - centroids[assignments]
        quantization_error = float(np.mean(np.sum(diffs * diffs, axis=1)))
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.tolist(),
            "quantization_error": quantization_error,
        }

def _kmeans_plus_plus_init(X: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """k-means++ initialization (numpy) using default_rng."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest_sq.sum())
        if total == 0.0:
            centers[i:] = centers[0]
            break
        probs = closest_sq / total
        r = float(rng.random())
        cum = np.cumsum(probs)
        idx = int(np.searchsorted(cum, r, side="right"))
        if idx >= n:
            idx = n - 1
        centers[i] = X[idx]
        dist_new = np.sum((X - centers[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_new)
    return centers

def _compute_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of centroids and counts using np.add.at.
    Returns (centers (k,d) float32, counts (k,) int64).
    """
    n, d = X.shape
    sums = np.zeros((k, d), dtype=np.float64)
    # accumulate sums per label
    np.add.at(sums, labels, X.astype(np.float64))
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    centers = np.zeros((k, d), dtype=np.float32)
    nonzero = counts > 0
    if np.any(nonzero):
        centers[nonzero] = (sums[nonzero] / counts[nonzero, None]).astype(np.float32)
    return centers, counts

def _kmeans_numpy(
    X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """A basic Lloyd's algorithm with k-means++ init. Returns (centroids, labels)."""
    n, d = X.shape
    centers = _kmeans_plus_plus_init(X, k, random_state=random_state).astype(np.float32)
    X_norm2 = np.sum(X * X, axis=1)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        centers_norm2 = np.sum(centers * centers, axis=1)
        # squared distances: ||x||^2 - 2 x·c + ||c||^2
        dists = X_norm2[:, None] - 2.0 * X.dot(centers.T) + centers_norm2[None, :]
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        old_centers = centers.copy()

        # Vectorized centroid computation
        centers, counts = _compute_centroids(X, labels, k)

        # reseed empty clusters if any
        if np.any(counts == 0):
            for i in np.where(counts == 0)[0]:
                centers[i] = X[np.random.randint(0, n)]

        shifts = np.sum((centers - old_centers) ** 2, axis=1)
        if np.max(shifts) <= tol * tol:
            break

    sizes = np.bincount(labels, minlength=k)
    if np.any(sizes == 0):
        centers, labels = _fix_empty_clusters(X, centers, labels)
    return centers.astype(np.float32), labels.astype(int)

def _fix_empty_clusters(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repair empty clusters by taking points farthest from their assigned centroids
    and using them to seed empty clusters, then run a few Lloyd iterations.
    Ensures every cluster has at least one point.
    """
    n, d = X.shape
    k = centers.shape[0]
    centers = centers.copy().astype(np.float32)
    labels = labels.copy().astype(int)
    X_norm2 = np.sum(X * X, axis=1)

    for attempt in range(10):
        counts = np.bincount(labels, minlength=k)
        empty = np.where(counts == 0)[0]
        if empty.size == 0:
            break
        centers_norm2 = np.sum(centers * centers, axis=1)
        dists = X_norm2[:, None] - 2.0 * X.dot(centers.T) + centers_norm2[None, :]
        for e in empty:
            counts = np.bincount(labels, minlength=k)
            largest = int(np.argmax(counts))
            pts_idx = np.where(labels == largest)[0]
            if pts_idx.size == 0:
                idx = np.random.randint(0, n)
            else:
                local_dists = np.sum((X[pts_idx] - centers[largest]) ** 2, axis=1)
                far_pos = int(np.argmax(local_dists))
                idx = int(pts_idx[far_pos])
            labels[idx] = e
            centers[e] = X[idx]
        # small stabilization: a few Lloyd iterations (vectorized)
        for _ in range(3):
            centers_norm2 = np.sum(centers * centers, axis=1)
            dists = X_norm2[:, None] - 2.0 * X.dot(centers.T) + centers_norm2[None, :]
            labels = np.argmin(dists, axis=1)
            centers, counts = _compute_centroids(X, labels, k)
            # reseed any still-empty
            if np.any(counts == 0):
                for i in np.where(counts == 0)[0]:
                    centers[i] = X[np.random.randint(0, n)]

    # final fallback: ensure no empty clusters
    counts = np.bincount(labels, minlength=k)
    for i in range(k):
        if counts[i] == 0:
            centers[i] = X[np.random.randint(0, n)]
            d = np.sum((X - centers[i]) ** 2, axis=1)
            labels[np.argmin(d)] = i
            counts = np.bincount(labels, minlength=k)
    return centers.astype(np.float32), labels.astype(int)