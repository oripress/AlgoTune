from typing import Any, List
import numpy as np

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[int]:
        """
        K-means clustering solver:
          - k-means++ initialization
          - vectorized or batched distance computations
          - multiple random restarts with early stopping
          - empty-cluster handling

        Returns:
            list[int]: labels for each sample (0..k-1)
        """
        try:
            # Parse input
            if isinstance(problem, dict):
                X = np.asarray(problem.get("X", []), dtype=np.float64)
                k = int(problem.get("k", 0))
            else:
                X = np.asarray(problem, dtype=np.float64)
                k = int(kwargs.get("k", 0))

            if X.size == 0:
                return []
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n, d = X.shape

            # Edge cases
            if k <= 0:
                return [0] * n
            if k == 1:
                return [0] * n
            if k >= n:
                # assign distinct labels 0..n-1 (all < k)
                return list(range(n))

            # Heuristics for number of inits and iterations
            if n <= 2000:
                n_init = 10
                max_iter = 300
            elif n <= 10000:
                n_init = 6
                max_iter = 200
            elif n <= 50000:
                n_init = 3
                max_iter = 150
            else:
                n_init = 1
                max_iter = 100

            rng = np.random.default_rng(0)

            # Precompute squared norms of X
            X_norm = np.einsum("ij,ij->i", X, X)

            # Memory heuristic for batching
            MAX_ENTRIES = 4_000_000
            # batch_size defined unconditionally to avoid used-before-assignment warnings
            batch_size = max(128, int(MAX_ENTRIES // max(1, k)))
            batch_size = min(batch_size, n)
            use_full = (n * k) <= MAX_ENTRIES
            if use_full:
                # when using full, batch_size won't be used, but keep defined
                batch_size = n

            def kmeans_pp_init():
                centers = np.empty((k, d), dtype=np.float64)
                centers_norm = np.empty(k, dtype=np.float64)

                # choose first center uniformly
                first = int(rng.integers(0, n))
                centers[0] = X[first]
                centers_norm[0] = centers[0].dot(centers[0])

                # distances to nearest chosen center (squared)
                closest = X_norm - 2.0 * X.dot(centers[0]) + centers_norm[0]
                np.maximum(closest, 0, out=closest)

                for i in range(1, k):
                    total = float(closest.sum())
                    if total <= 0.0 or not np.isfinite(total):
                        idx = int(rng.integers(0, n))
                    else:
                        probs = closest / total
                        # rng.choice with p can fail if probs contain nans; fallback to uniform
                        try:
                            idx = int(rng.choice(n, p=probs))
                        except Exception:
                            idx = int(rng.integers(0, n))
                    centers[i] = X[idx]
                    centers_norm[i] = centers[i].dot(centers[i])
                    dist = X_norm - 2.0 * X.dot(centers[i]) + centers_norm[i]
                    np.maximum(dist, 0, out=dist)
                    np.minimum(closest, dist, out=closest)
                return centers

            def assign_labels_and_dists(centers):
                centers_norm_local = np.einsum("ij,ij->i", centers, centers)
                labels = np.empty(n, dtype=np.int64)
                dists = np.empty(n, dtype=np.float64)
                if use_full:
                    # compute full distance matrix
                    dots = X.dot(centers.T)
                    distances = X_norm[:, None] - 2.0 * dots + centers_norm_local[None, :]
                    np.maximum(distances, 0, out=distances)
                    labels = np.argmin(distances, axis=1).astype(np.int64)
                    dists = distances[np.arange(n), labels]
                else:
                    for s in range(0, n, batch_size):
                        e = min(n, s + batch_size)
                        dots = X[s:e].dot(centers.T)
                        block = X_norm[s:e, None] - 2.0 * dots + centers_norm_local[None, :]
                        np.maximum(block, 0, out=block)
                        lb = np.argmin(block, axis=1).astype(np.int64)
                        labels[s:e] = lb
                        dists[s:e] = block[np.arange(e - s), lb]
                return labels, dists

            best_inertia = float("inf")
            best_labels: List[int] = [0] * n

            for _init in range(n_init):
                centers = kmeans_pp_init()
                prev_labels = np.full(n, -1, dtype=np.int64)

                for _it in range(max_iter):
                    labels, dists = assign_labels_and_dists(centers)
                    if np.array_equal(labels, prev_labels):
                        break
                    prev_labels = labels.copy()

                    counts = np.bincount(labels, minlength=k)

                    # compute sums per cluster using np.add.at (efficient)
                    sums = np.zeros((k, d), dtype=np.float64)
                    np.add.at(sums, labels, X)

                    nonzero = counts > 0
                    if nonzero.any():
                        centers[nonzero] = sums[nonzero] / counts[nonzero][:, None]

                    # handle empty clusters by placing them at farthest points
                    if not nonzero.all():
                        cur_dists = dists.copy()
                        empty_idx = np.where(counts == 0)[0]
                        for j in empty_idx:
                            # pick the point with largest distance to its assigned center
                            idx = int(np.argmax(cur_dists))
                            centers[j] = X[idx]
                            # ensure we don't pick the same point again
                            cur_dists[idx] = -np.inf
                            counts[j] = 1

                final_labels, final_dists = assign_labels_and_dists(centers)
                inertia = float(final_dists.sum())

                if inertia + 1e-12 < best_inertia:
                    best_inertia = inertia
                    best_labels = final_labels.tolist()
                    if best_inertia <= 1e-12:
                        break

            return best_labels

        except Exception:
            # Fallback: trivial labeling
            try:
                n_val = int(np.asarray(problem.get("X", [])).shape[0])
            except Exception:
                try:
                    n_val = len(problem.get("X", []) or [])
                except Exception:
                    n_val = 0
            return [0] * n_val