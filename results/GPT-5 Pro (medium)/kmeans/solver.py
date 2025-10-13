from __future__ import annotations

from typing import Any, Optional

import numpy as np


try:
    # Optional dependency; used if available
    from sklearn.cluster import KMeans as _SKKMeans  # type: ignore
except Exception:  # pragma: no cover
    _SKKMeans = None  # type: ignore


def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)

    # Pick the first center randomly
    i0 = int(rng.integers(n))
    centers[0] = X[i0]

    # Initialize closest squared distances
    closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

    for c in range(1, k):
        total = float(closest_dist_sq.sum())
        if not np.isfinite(total) or total <= 0.0:
            # All points are identical (or numeric issues); pick random point
            idx = int(rng.integers(n))
            centers[c] = X[idx]
        else:
            probs = closest_dist_sq / total
            idx = int(rng.choice(n, p=probs))
            centers[c] = X[idx]
            # Update closest distances
            dist_new = np.sum((X - centers[c]) ** 2, axis=1)
            np.minimum(closest_dist_sq, dist_new, out=closest_dist_sq)

    return centers


def _lloyd_iterations(
    X: np.ndarray,
    init_centers: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Lloyd's algorithm starting from init_centers.
    Returns (labels, centers, inertia)."""
    n, _ = X.shape
    centers = init_centers.copy()
    k = centers.shape[0]

    X_norm2 = np.einsum("ij,ij->i", X, X)

    last_shift = np.inf
    for _ in range(max_iter):
        C_norm2 = np.einsum("ij,ij->i", centers, centers)
        # Squared distances using (x-c)^2 = ||x||^2 + ||c||^2 - 2 x.c
        distances = X_norm2[:, None] + C_norm2[None, :] - 2.0 * (X @ centers.T)

        labels = distances.argmin(axis=1)
        # Compute new centers
        new_centers = np.zeros_like(centers)
        counts = np.bincount(labels, minlength=k)
        for j in range(k):
            if counts[j] > 0:
                new_centers[j] = X[labels == j].mean(axis=0)
            else:
                # Reinitialize empty cluster to a random data point
                idx = int(rng.integers(n))
                new_centers[j] = X[idx]

        shift = float(np.sum((centers - new_centers) ** 2))
        centers = new_centers

        if not np.isfinite(shift):
            break

        # Convergence check relative to center magnitude
        denom = float(np.sum(centers ** 2)) + 1e-12
        if shift / denom <= tol or abs(last_shift - shift) <= tol * denom:
            break
        last_shift = shift

    # Final assignment and inertia
    C_norm2 = np.einsum("ij,ij->i", centers, centers)
    distances = X_norm2[:, None] + C_norm2[None, :] - 2.0 * (X @ centers.T)
    labels = distances.argmin(axis=1)
    inertia = float(np.sum(distances[np.arange(n), labels]))
    return labels, centers, inertia


def _kmeans_numpy(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    n_init: int = 5,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """NumPy-only KMeans with k-means++ init as a fallback if sklearn is unavailable."""
    rng = np.random.default_rng(random_state)
    best_labels: Optional[np.ndarray] = None
    best_inertia = np.inf

    for _ in range(max(1, n_init)):
        centers0 = _kmeanspp_init(X, k, rng)
        labels, centers, inertia = _lloyd_iterations(X, centers0, rng, max_iter=max_iter, tol=tol)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    assert best_labels is not None
    return best_labels


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[int]:
        X = problem.get("X", None)
        k = int(problem.get("k", 0))

        if X is None:
            return []

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        if n == 0:
            return []

        # Trivial fast paths
        if k <= 1:
            return [0] * n
        if k >= n:
            # Each point becomes its own cluster (optimal inertia 0)
            return list(range(n))

        # Prefer sklearn's highly optimized implementation if available
        if _SKKMeans is not None:
            # Try with Elkan algorithm for speed on dense data; gracefully
            # fall back if parameter is unsupported in the installed version.
            for params in (
                dict(n_clusters=k, n_init=10, algorithm="elkan"),
                dict(n_clusters=k, n_init=10),
                dict(n_clusters=k),
            ):
                try:
                    model = _SKKMeans(**params)  # type: ignore[arg-type]
                    labels = model.fit_predict(X)
                    return labels.tolist()
                except TypeError:
                    # Try next parameter combination (API differences across versions)
                    continue
                except Exception:
                    # If fit failed due to numerical issues, try next or fallback
                    break

        # Fallback: NumPy implementation (good quality, reasonably fast)
        labels_np = _kmeans_numpy(X, k, max_iter=100, n_init=min(5, max(1, k)), tol=1e-4, random_state=0)
        return labels_np.tolist()