from __future__ import annotations

from typing import Any, Dict, Tuple
import os

import numpy as np
import faiss

# Configure FAISS threading once at import time (avoid per-call overhead)
try:
    if hasattr(faiss, "omp_set_num_threads"):
        env_threads = os.environ.get("FAISS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
        threads = int(env_threads) if env_threads is not None else (os.cpu_count() or 1)
        if threads and threads > 0:
            faiss.omp_set_num_threads(int(threads))
except Exception:
    pass

class Solver:
    def _assign_to_centroids(self, vectors: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assign vectors to nearest centroids using FAISS brute-force knn if available, else IndexFlatL2.

        Returns:
            distances: (n, 1) array of squared L2 distances to nearest centroid
            assignments: (n,) array of assigned centroid indices
        """
        # Ensure float32 C-contiguous
        vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        centroids = np.ascontiguousarray(centroids.astype(np.float32, copy=False))

        if hasattr(faiss, "knn"):
            # Use optimized brute-force KNN (often faster than building an index)
            distances, assignments = faiss.knn(vectors, centroids, 1)
            return distances, assignments.reshape(-1)

        # Fallback to IndexFlatL2
        dim = centroids.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)
        return distances, assignments.reshape(-1)

        # Note: distances are squared L2

    def _fix_empty_clusters(
        self,
        vectors: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
        distances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ensure no clusters are empty by moving farthest points to empty clusters."""
        k = centroids.shape[0]
        # Count current cluster sizes
        cluster_sizes = np.bincount(assignments, minlength=k)
        empty = np.flatnonzero(cluster_sizes == 0)
        if empty.size == 0:
            return centroids, assignments, distances

        # Flatten distances to (n,)
        dists = distances.reshape(-1)

        # We will iteratively fix empties; cap iterations to avoid pathological loops
        max_attempts = 3
        attempt = 0
        # Use a working copy
        work_centroids = centroids.copy()
        work_assignments = assignments.copy()
        work_distances = dists.copy()

        while attempt < max_attempts:
            cluster_sizes = np.bincount(work_assignments, minlength=k)
            empty = np.flatnonzero(cluster_sizes == 0)
            if empty.size == 0:
                break

            # Valid candidate points are those whose current cluster has size > 1
            valid_mask = cluster_sizes[work_assignments] > 1
            valid_indices = np.flatnonzero(valid_mask)
            if valid_indices.size == 0:
                # Cannot fix empties safely; fall back to assigning random points (still ensures non-empty)
                rng = np.random.default_rng()
                choices = rng.choice(vectors.shape[0], size=empty.size, replace=(vectors.shape[0] < empty.size))
                for e_idx, vec_idx in zip(empty, np.unique(choices)):
                    work_centroids[e_idx] = vectors[vec_idx]
                    work_assignments[vec_idx] = e_idx
                # Re-assign all points to nearest centroids
                dist2, assign2 = self._assign_to_centroids(vectors, work_centroids)
                work_assignments = assign2.astype(np.int64, copy=False)
                work_distances = dist2.reshape(-1)
                attempt += 1
                continue

            # Pick 'm' farthest valid points (m = number of empty clusters)
            m = empty.size
            valid_dists = work_distances[valid_indices]
            if valid_indices.size <= m:
                farthest_indices = valid_indices  # take all
            else:
                # Argpartition for top-m
                part = np.argpartition(valid_dists, -m)[-m:]
                # Sort these for determinism (optional)
                order = np.argsort(valid_dists[part])[::-1]
                farthest_indices = valid_indices[part[order]]

            # Assign each selected point to one empty cluster and set centroid to that vector
            if farthest_indices.size < m:
                needed = m - farthest_indices.size
                remaining_candidates = np.setdiff1d(valid_indices, farthest_indices, assume_unique=False)
                if remaining_candidates.size == 0:
                    remaining_candidates = valid_indices
                rng = np.random.default_rng()
                picks = rng.choice(remaining_candidates, size=needed, replace=(remaining_candidates.size < needed))
                farthest_indices = np.concatenate([farthest_indices, np.unique(picks)])

            for e_idx, vec_idx in zip(empty, farthest_indices[:m]):
                work_centroids[e_idx] = vectors[vec_idx]
                work_assignments[vec_idx] = e_idx

            # Re-assign all points to nearest centroids with updated centroids
            dist2, assign2 = self._assign_to_centroids(vectors, work_centroids)
            work_assignments = assign2.astype(np.int64, copy=False)
            work_distances = dist2.reshape(-1)

            attempt += 1

        # Final safety: if still empty (extremely unlikely), force assign arbitrary distinct points
        cluster_sizes = np.bincount(work_assignments, minlength=k)
        empty = np.flatnonzero(cluster_sizes == 0)
        if empty.size > 0:
            all_indices = np.arange(vectors.shape[0])
            size_per_point = cluster_sizes[work_assignments]
            order = np.argsort(size_per_point)[::-1]
            chosen = []
            taken_points = set()
            for idx in order:
                if len(chosen) == empty.size:
                    break
                if size_per_point[idx] > 1 and idx not in taken_points:
                    chosen.append(idx)
                    taken_points.add(idx)
            if len(chosen) < empty.size:
                remaining = [i for i in all_indices if i not in taken_points]
                chosen += remaining[: (empty.size - len(chosen))]
            for e_idx, vec_idx in zip(empty, chosen[: empty.size]):
                work_centroids[e_idx] = vectors[vec_idx]
                work_assignments[vec_idx] = e_idx
            dist2, assign2 = self._assign_to_centroids(vectors, work_centroids)
            work_assignments = assign2.astype(np.int64, copy=False)
            work_distances = dist2.reshape(-1)

        return work_centroids, work_assignments, work_distances.reshape(-1, 1)

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the Vector Quantization problem using FAISS k-means with safeguards.

        Returns:
            - centroids: ndarray of shape (k, dim)
            - assignments: ndarray of shape (n,)
            - quantization_error: mean squared distance to nearest centroid (float)
        """
        vectors_in = problem["vectors"]
        k = int(problem["k"])
        # Convert to contiguous float32 for faiss
        vectors = np.asarray(vectors_in, dtype=np.float32)
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)
        n, dim = vectors.shape

        # Fast trivial cases
        if k <= 1:
            centroid = vectors.mean(axis=0, dtype=np.float64, keepdims=False).astype(np.float32, copy=False)
            diffs = vectors - centroid  # broadcasting
            mse = float(np.square(diffs).sum(axis=1).mean())
            return {
                "centroids": centroid.reshape(1, -1),
                "assignments": np.zeros(n, dtype=np.int64),
                "quantization_error": mse,
            }

        if k >= n:
            # Each vector can be its own centroid (and duplicates if k > n)
            if k == n:
                centroids = vectors.copy()
                assignments = np.arange(n, dtype=np.int64)
                quantization_error = 0.0
                return {
                    "centroids": centroids,
                    "assignments": assignments,
                    "quantization_error": quantization_error,
                }
            # k > n: repeat some vectors as centroids (not typically used in eval)
            reps = int(np.ceil(k / n))
            tiled_idx = np.tile(np.arange(n), reps)[:k]
            centroids = vectors[tiled_idx].copy()
            distances, assignments = self._assign_to_centroids(vectors, centroids)
            assignments = assignments.astype(np.int64, copy=False)
            qerr = float(distances.mean())
            return {
                "centroids": centroids,
                "assignments": assignments,
                "quantization_error": qerr,
            }

        # Train k-means using FAISS (defaults chosen to match reference quality)
        try:
            kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        except TypeError:
            kmeans = faiss.Kmeans(dim, k)

        kmeans.train(vectors)
        centroids = np.asarray(kmeans.centroids, dtype=np.float32)
        if not centroids.flags["C_CONTIGUOUS"]:
            centroids = np.ascontiguousarray(centroids)

        # Assign vectors to nearest centroids using fast faiss.knn (preferred) or IndexFlatL2 fallback
        distances, assignments = self._assign_to_centroids(vectors, centroids)
        assignments = assignments.astype(np.int64, copy=False)

        # Check for empty clusters; fix only if necessary
        cluster_sizes = np.bincount(assignments, minlength=k)
        if np.any(cluster_sizes == 0):
            centroids, assignments, distances = self._fix_empty_clusters(vectors, centroids, assignments, distances)

        # Compute quantization error as mean squared distance
        quantization_error = float(distances.mean())

        # Prepare output (as numpy arrays for minimal conversion overhead)
        return {
            "centroids": centroids,
            "assignments": assignments,
            "quantization_error": quantization_error,
        }