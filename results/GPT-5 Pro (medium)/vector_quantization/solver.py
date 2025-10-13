from typing import Any

import os
import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Vector Quantization problem using Faiss KMeans with optimized assignments.

        Parameters:
            problem (dict): {
                "n_vectors": int,
                "dim": int,
                "k": int,
                "vectors": List[List[float]]
            }

        Returns:
            dict: {
                "centroids": List[List[float]],
                "assignments": List[int],
                "quantization_error": float
            }
        """
        vectors = np.asarray(problem["vectors"], dtype=np.float32)
        k = int(problem["k"])
        if vectors.ndim != 2:
            vectors = vectors.reshape(-1, vectors.shape[-1])
        n, dim = vectors.shape

        # Ensure faiss uses all available threads
        try:
            n_threads = os.cpu_count() or 1
            faiss.omp_set_num_threads(max(1, n_threads))
        except Exception:
            pass

        # Handle trivial/special cases
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if k > n:
            # Assuming test inputs satisfy k <= n, as non-empty cluster validation is enforced.
            k = n

        # Fast path: k == 1 -> centroid is mean
        if k == 1:
            centroid = vectors.mean(axis=0, dtype=np.float32, keepdims=True)
            diffs = vectors - centroid
            dists = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
            return {
                "centroids": centroid.tolist(),
                "assignments": [0] * n,
                "quantization_error": float(dists.mean()),
            }

        # Fast path: k == n -> each vector its own centroid (zero error)
        if k == n:
            # Each point is its own centroid
            centroids = vectors.copy()
            assignments = np.arange(n, dtype=np.int32)
            return {
                "centroids": centroids.tolist(),
                "assignments": assignments.tolist(),
                "quantization_error": 0.0,
            }

        # Train KMeans with Faiss
        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        try:
            # Ensure at least one point per centroid during training
            kmeans.cp.min_points_per_centroid = 1  # type: ignore[attr-defined]
        except Exception:
            pass

        kmeans.train(vectors)
        centroids = kmeans.centroids.astype(np.float32, copy=False)

        # Optimized assignment using GEMM-based distances with batching to limit memory
        X = vectors
        C = centroids
        x_norm = np.einsum("ij,ij->i", X, X, dtype=np.float32)
        c_norm = np.einsum("ij,ij->i", C, C, dtype=np.float32)

        n, d = X.shape
        k = C.shape[0]

        # Choose strategy based on memory footprint
        # Limit to about 10 million float32 elements (~40MB) for the temporary distance matrix per batch
        max_elems = 10_000_000
        use_batched = n * k > max_elems

        assignments = np.empty(n, dtype=np.int32)
        min_dists = np.empty(n, dtype=np.float32)

        if not use_batched:
            # Single-shot computation
            # dist^2 = ||x||^2 + ||c||^2 - 2 x @ c^T
            # Compute -2 x @ c^T
            prod = X @ C.T  # float32 GEMM
            d2 = (x_norm[:, None] + c_norm[None, :] - 2.0 * prod).astype(np.float32, copy=False)
            # Numerical stability
            np.maximum(d2, 0.0, out=d2)
            assignments[:] = np.argmin(d2, axis=1)
            min_dists[:] = d2[np.arange(n), assignments]
        else:
            # Batched computation
            # Determine chunk size: chunk_size * k <= max_elems
            chunk_size = max(1, max_elems // max(k, 1))
            CT = C.T  # reuse transpose across batches
            start = 0
            while start < n:
                end = min(n, start + chunk_size)
                Xb = X[start:end]
                prod = Xb @ CT
                d2 = (x_norm[start:end, None] + c_norm[None, :] - 2.0 * prod).astype(np.float32, copy=False)
                np.maximum(d2, 0.0, out=d2)
                a = np.argmin(d2, axis=1).astype(np.int32, copy=False)
                assignments[start:end] = a
                min_dists[start:end] = d2[np.arange(end - start), a]
                start = end

        quantization_error = float(min_dists.mean())

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.tolist(),
            "quantization_error": quantization_error,
        }