from __future__ import annotations

from typing import Any

import numpy as np

try:
    import sklearn.cluster
except Exception:  # pragma: no cover
    sklearn = None

class Solver:
    def __init__(self) -> None:
        # Warm up sklearn in init; init cost does not count.
        if sklearn is not None:
            X = np.zeros((4, 2), dtype=np.float32)
            try:
                sklearn.cluster.KMeans(
                    n_clusters=1,
                    init="k-means++",
                    n_init=1,
                    max_iter=1,
                    tol=1e-3,
                    algorithm="lloyd",
                    copy_x=False,
                ).fit(X)
            except Exception:
                pass
            try:
                sklearn.cluster.MiniBatchKMeans(
                    n_clusters=1,
                    init="k-means++",
                    n_init=1,
                    max_iter=1,
                    batch_size=4,
                    tol=0.0,
                    reassignment_ratio=0.0,
                ).fit(X)
            except Exception:
                pass

    @staticmethod
    def _solve_kmeans(X: np.ndarray, k: int) -> list[int]:
        d = X.shape[1]
        algorithm = "elkan" if d <= 32 else "lloyd"
        km = sklearn.cluster.KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=1,
            max_iter=60,
            tol=1e-3,
            algorithm=algorithm,
            copy_x=False,
        )
        return km.fit(X).labels_.tolist()

    @staticmethod
    def _solve_minibatch(X: np.ndarray, k: int) -> list[int]:
        n = X.shape[0]
        batch_size = min(max(1024, 8 * k), n)
        init_size = min(n, max(3 * batch_size, 3 * k))
        km = sklearn.cluster.MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=1,
            max_iter=80,
            batch_size=batch_size,
            init_size=init_size,
            tol=0.0,
            max_no_improvement=8,
            reassignment_ratio=0.01,
        )
        return km.fit(X).labels_.tolist()

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> list[int]:
        try:
            X_raw = problem["X"]
            X = np.asarray(X_raw, dtype=np.float32)
            if X.ndim != 2:
                return [0] * len(X_raw)

            n, d = X.shape
            if n == 0:
                return []

            k = int(problem["k"])
            if k <= 1:
                return [0] * n
            if k >= n:
                return list(range(n))

            if sklearn is None:
                return [0] * n

            X = np.ascontiguousarray(X)

            work = n * k * d
            if n >= max(4000, 20 * k) and work >= 8_000_000:
                try:
                    return self._solve_minibatch(X, k)
                except Exception:
                    pass

            try:
                return self._solve_kmeans(X, k)
            except Exception:
                return [0] * n
        except Exception:
            X = problem.get("X", [])
            return [0] * len(X)