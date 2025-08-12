from __future__ import annotations

from typing import Any, Tuple
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[int]:
        try:
            X = np.asarray(problem["X"])
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n, d = X.shape
            k = int(problem["k"])

            # Edge cases
            if n == 0:
                return []
            if k <= 1:
                return [0] * n
            if k >= n:
                # Assign each point to a unique cluster; remaining clusters unused.
                return list(range(n))

            # Prefer float32 contiguous for speed/memory
            if X.dtype != np.float32:
                X = X.astype(np.float32, copy=False)
            if not X.flags.c_contiguous:
                X = np.ascontiguousarray(X)

            # Seed handling
            seed = int(kwargs.get("seed", 42))

            size_scale = n * d
            nk = n * k

            # For very small problems, a lightweight NumPy Lloyd's can beat FAISS overhead
            if size_scale <= 8192 and nk <= 4_000_000:
                return _kmeans_numpy_small(X, k, seed)

            # For moderate sizes and small k, a mid-size NumPy KMeans is faster than FAISS
            if k <= 32 and size_scale <= 2_000_000:
                return _kmeans_numpy_fast(X, k, seed)

            # Try FAISS KMeans for speed
            try:
                import faiss  # type: ignore

                try:
                    faiss.cvar.rand_seed = seed  # set global seed if available
                except Exception:
                    pass

                # Further reduced iterations/redos for speed while maintaining quality
                if size_scale <= 50_000:
                    niter = 16
                    nredo = 1
                elif size_scale <= 500_000:
                    niter = 14
                    nredo = 1
                elif size_scale <= 2_000_000:
                    niter = 12
                    nredo = 1
                else:
                    niter = 10
                    nredo = 1

                kmeans = faiss.Kmeans(
                    d,
                    k,
                    niter=niter,
                    nredo=nredo,
                    verbose=False,
                    spherical=False,
                    min_points_per_centroid=1,
                )
                # Tune clustering parameters to reduce training cost
                try:
                    kmeans.cp.seed = seed  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    # ensure at least some samples per centroid and cap to avoid huge work
                    avg_per_centroid = max(1, n // max(k, 1))
                    mppc = int(max(32, min(512, avg_per_centroid)))
                    kmeans.cp.max_points_per_centroid = mppc  # type: ignore[attr-defined]
                except Exception:
                    pass

                kmeans.train(X)

                # Assign points to nearest centroid using the built-in index
                _, I = kmeans.index.search(X, 1)
                return I.ravel().astype(np.int32, copy=False).tolist()

            except Exception:
                # Fallback to scikit-learn's optimized KMeans (elkan) if FAISS unavailable
                try:
                    from sklearn.cluster import KMeans  # type: ignore

                    km = KMeans(
                        n_clusters=k,
                        n_init=2,
                        algorithm="elkan",
                        random_state=seed,
                    ).fit(X)
                    return km.labels_.astype(np.int32, copy=False).tolist()
                except Exception:
                    # Final fallback: NumPy implementation (k-means++ + Lloyd)
                    pass

            # NumPy implementation fallback (more robust, slightly heavier than small variant)
            rng = np.random.default_rng(seed)

            # Precompute norms for distance computations
            x_norms = np.einsum("ij,ij->i", X, X)

            # Adaptive parameters
            if size_scale <= 50_000:
                n_init = 3
                max_iter = 32
            elif size_scale <= 500_000:
                n_init = 2
                max_iter = 28
            elif size_scale <= 2_000_000:
                n_init = 2
                max_iter = 24
            else:
                n_init = 1
                max_iter = 22

            rel_tol = 1e-3  # relative improvement threshold for early stopping

            best_labels: np.ndarray | None = None
            best_inertia = np.inf

            for _ in range(n_init):
                centers = _kmeanspp_init(X, k, rng, x_norms=x_norms)

                prev_labels: np.ndarray | None = None
                prev_inertia = np.inf
                min_dists = None  # type: ignore
                for _ in range(max_iter):
                    labels, min_dists = _assign_labels(X, centers, x_norms)
                    inertia = float(np.sum(min_dists))
                    # Early stopping: if labels stable or small relative improvement
                    if prev_labels is not None:
                        if np.array_equal(labels, prev_labels):
                            break
                        if prev_inertia - inertia <= rel_tol * max(prev_inertia, 1.0):
                            break
                    prev_labels = labels
                    prev_inertia = inertia

                    counts = np.bincount(labels, minlength=k).astype(np.int64, copy=False)
                    # Select mean computation based on dimensionality
                    if X.shape[1] <= 64:
                        new_centers = _compute_means_fast(X, labels, counts, k)
                    else:
                        new_centers = _compute_means(X, labels, counts, k)

                    # Handle empty clusters: re-seed them to farthest points
                    empties = np.where(counts == 0)[0]
                    if empties.size > 0:
                        # Use farthest points (by current assignment distances)
                        farthest_idx = np.argpartition(min_dists, -empties.size)[-empties.size:]
                        new_centers[empties] = X[farthest_idx]

                    centers = new_centers

                # Final inertia for this init
                if min_dists is None or prev_labels is None:
                    labels, min_dists = _assign_labels(X, centers, x_norms)
                    inertia = float(np.sum(min_dists))
                else:
                    inertia = float(np.sum(min_dists))

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_labels = labels.copy()

            assert best_labels is not None
            return best_labels.astype(np.int32, copy=False).tolist()

        except Exception:
            n = 0
            try:
                n = len(problem["X"])  # type: ignore[index]
            except Exception:
                pass
            return [0] * n

def _kmeans_numpy_fast(X: np.ndarray, k: int, seed: int) -> list[int]:
    """Mid-size fast NumPy KMeans for small k, moderate n*d."""
    n, d = X.shape
    size_scale = n * d
    rng = np.random.default_rng(seed)
    x_norms = np.einsum("ij,ij->i", X, X)

    if size_scale <= 200_000:
        n_init = 2
        max_iter = 20
    elif size_scale <= 1_000_000:
        n_init = 2
        max_iter = 18
    else:
        n_init = 1
        max_iter = 16

    rel_tol = 1e-3

    best_labels: np.ndarray | None = None
    best_inertia = np.inf
    for _ in range(n_init):
        centers = _kmeanspp_init(X, k, rng, x_norms=x_norms)

        prev_labels: np.ndarray | None = None
        prev_inertia = np.inf
        min_dists = None  # type: ignore
        for _ in range(max_iter):
            labels, min_dists = _assign_labels(X, centers, x_norms)
            inertia = float(np.sum(min_dists))
            if prev_labels is not None:
                if np.array_equal(labels, prev_labels):
                    break
                if prev_inertia - inertia <= rel_tol * max(prev_inertia, 1.0):
                    break
            prev_labels = labels
            prev_inertia = inertia

            counts = np.bincount(labels, minlength=k).astype(np.int64, copy=False)
            if d <= 64:
                new_centers = _compute_means_fast(X, labels, counts, k)
            else:
                new_centers = _compute_means(X, labels, counts, k)

            empties = np.where(counts == 0)[0]
            if empties.size > 0:
                farthest_idx = np.argpartition(min_dists, -empties.size)[-empties.size:]
                new_centers[empties] = X[farthest_idx]

            centers = new_centers

        if min_dists is None or prev_labels is None:
            labels, min_dists = _assign_labels(X, centers, x_norms)
            inertia = float(np.sum(min_dists))
        else:
            inertia = float(np.sum(min_dists))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    assert best_labels is not None
    return best_labels.astype(np.int32, copy=False).tolist()

def _kmeans_numpy_small(X: np.ndarray, k: int, seed: int) -> list[int]:
    """Lightweight KMeans for small problems to avoid FAISS overhead."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    x_norms = np.einsum("ij,ij->i", X, X)

    # Conservative but small params
    n_init = 1
    max_iter = 14
    rel_tol = 1e-3

    best_labels: np.ndarray | None = None
    best_inertia = np.inf

    for _ in range(n_init):
        centers = _kmeanspp_init(X, k, rng, x_norms=x_norms)

        prev_labels: np.ndarray | None = None
        prev_inertia = np.inf
        min_dists = None  # type: ignore
        for _ in range(max_iter):
            labels, min_dists = _assign_labels_small(X, centers, x_norms)
            inertia = float(np.sum(min_dists))
            if prev_labels is not None:
                if np.array_equal(labels, prev_labels):
                    break
                if prev_inertia - inertia <= rel_tol * max(prev_inertia, 1.0):
                    break
            prev_labels = labels
            prev_inertia = inertia

            counts = np.bincount(labels, minlength=k).astype(np.int64, copy=False)
            new_centers = _compute_means_fast(X, labels, counts, k)

            # Handle empty clusters by snapping to farthest points
            empties = np.where(counts == 0)[0]
            if empties.size > 0:
                farthest_idx = np.argpartition(min_dists, -empties.size)[-empties.size:]
                new_centers[empties] = X[farthest_idx]

            centers = new_centers

        if min_dists is None or prev_labels is None:
            labels, min_dists = _assign_labels_small(X, centers, x_norms)
            inertia = float(np.sum(min_dists))
        else:
            inertia = float(np.sum(min_dists))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    assert best_labels is not None
    return best_labels.astype(np.int32, copy=False).tolist()

def _kmeanspp_init(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    x_norms: np.ndarray | None = None,
) -> np.ndarray:
    """k-means++ initialization."""
    n, d = X.shape
    if x_norms is None:
        x_norms = np.einsum("ij,ij->i", X, X)

    centers = np.empty((k, d), dtype=X.dtype)
    # Choose first center uniformly at random
    first = int(rng.integers(n))
    centers[0] = X[first]

    # Distances to the nearest chosen center
    c = centers[0]
    c_norm = float(np.dot(c, c))
    # Compute squared distances to first center
    dists = x_norms + c_norm - 2.0 * (X @ c)
    np.maximum(dists, 0.0, out=dists)

    for i in range(1, k):
        total = float(dists.sum())
        if not np.isfinite(total) or total <= 1e-12:
            # Degenerate case: all points identical or numerically zero distances
            # Fill remaining centers with random points
            idx = int(rng.integers(n))
            centers[i] = X[idx]
            # Update dists with this center too, in case next iteration uses it
            c = centers[i]
            c_norm = float(np.dot(c, c))
            new_d = x_norms + c_norm - 2.0 * (X @ c)
            np.maximum(new_d, 0.0, out=new_d)
            np.minimum(dists, new_d, out=dists)
            continue

        # Sample new center index with probability proportional to D^2
        r = float(rng.random()) * total
        cumsum = np.cumsum(dists)
        j = int(np.searchsorted(cumsum, r, side="right"))
        if j >= n:
            j = n - 1
        centers[i] = X[j]

        # Update nearest distances
        c = centers[i]
        c_norm = float(np.dot(c, c))
        new_d = x_norms + c_norm - 2.0 * (X @ c)
        np.maximum(new_d, 0.0, out=new_d)
        np.minimum(dists, new_d, out=dists)

    return centers

def _assign_labels_small(
    X: np.ndarray,
    centers: np.ndarray,
    x_norms: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign labels for small problems with a single dense distance matrix."""
    if x_norms is None:
        x_norms = np.einsum("ij,ij->i", X, X)
    c_norms = np.einsum("ij,ij->i", centers, centers)
    # Dense distance matrix
    D = x_norms[:, None] + c_norms[None, :] - 2.0 * (X @ centers.T)
    labels = np.argmin(D, axis=1).astype(np.int32, copy=False)
    min_dists = D[np.arange(X.shape[0]), labels]
    np.maximum(min_dists, 0.0, out=min_dists)
    return labels, min_dists

def _assign_labels(
    X: np.ndarray,
    centers: np.ndarray,
    x_norms: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign labels to nearest centers and return labels and squared distances."""
    n = X.shape[0]
    k = centers.shape[0]
    if x_norms is None:
        x_norms = np.einsum("ij,ij->i", X, X)
    c_norms = np.einsum("ij,ij->i", centers, centers)

    labels = np.empty(n, dtype=np.int32)
    min_dists = np.empty(n, dtype=X.dtype)

    # Chunking to control memory usage of distance matrix
    # Aim for at most ~20M float32 elements in the (chunk_size x k) matrix
    # 20M float32 ~ 80MB
    max_elems = 20_000_000
    chunk_size = max(1, int(max_elems // max(k, 1)))
    if chunk_size >= n:
        chunk_size = n

    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        Xc = X[start:end]
        temp = Xc @ centers.T  # (m, k)
        D = (-2.0) * temp
        D += x_norms[start:end][:, None]
        D += c_norms[None, :]

        l = np.argmin(D, axis=1)
        labels[start:end] = l
        min_dists[start:end] = D[np.arange(end - start), l]
        start = end

    np.maximum(min_dists, 0.0, out=min_dists)
    return labels, min_dists

def _compute_means_fast(
    X: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute means using bincount across dimensions when beneficial."""
    n, d = X.shape
    centers = np.zeros((k, d), dtype=X.dtype)
    nonzero = counts > 0
    # Use bincount per dimension if d is small/moderate and k*d not too large
    if d <= 64 and (k * d) <= 2_000_000:
        for j in range(d):
            centers[:, j] = np.bincount(labels, weights=X[:, j], minlength=k)
    else:
        # Fallback accumulation
        np.add.at(centers, labels, X)
    if np.any(nonzero):
        centers[nonzero] /= counts[nonzero, None]
    return centers

def _compute_means(
    X: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute cluster means given labels and counts. counts length == k."""
    d = X.shape[1]
    centers = np.zeros((k, d), dtype=X.dtype)
    # Efficient accumulation: np.add.at performs indexed accumulation
    np.add.at(centers, labels, X)
    nonzero = counts > 0
    if np.any(nonzero):
        centers[nonzero] /= counts[nonzero, None]
    return centers