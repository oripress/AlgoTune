import os
from typing import Any

import faiss
import numpy as np

class Solver:
    @staticmethod
    def _to_f32_c(vectors: Any) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2:
            x = np.array(vectors, dtype=np.float32, copy=False)
            x = x.reshape(len(x), -1)
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        return x

    @staticmethod
    def _single_cluster(x: np.ndarray) -> dict[str, Any]:
        centroid = x.mean(axis=0, keepdims=True)
        diff = x - centroid
        distances = np.einsum("ij,ij->i", diff, diff)
        return {
            "centroids": centroid,
            "assignments": np.zeros(x.shape[0], dtype=np.int64),
            "quantization_error": float(distances.mean()),
        }

    @staticmethod
    def _identity_clusters(x: np.ndarray) -> dict[str, Any]:
        n = x.shape[0]
        return {
            "centroids": x.copy(),
            "assignments": np.arange(n, dtype=np.int64),
            "quantization_error": 0.0,
        }

    @staticmethod
    def _repair_empty(
        assignments: np.ndarray, distances: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.bincount(assignments, minlength=k)
        if np.all(counts > 0):
            return assignments, counts
        assignments = assignments.copy()
        d = distances.reshape(-1)
        for empty in np.flatnonzero(counts == 0):
            movable = counts[assignments] > 1
            idx = int(np.argmax(np.where(movable, d, -1.0)))
            old = int(assignments[idx])
            counts[old] -= 1
            assignments[idx] = empty
            counts[empty] = 1
            d[idx] = -1.0
        return assignments, counts

    @staticmethod
    def _means_and_mse(
        x: np.ndarray, assignments: np.ndarray, k: int, counts: np.ndarray
    ) -> tuple[np.ndarray, float]:
        centroids = np.empty((k, x.shape[1]), dtype=np.float32)
        denom = counts.astype(np.float32, copy=False)
        for j in range(x.shape[1]):
            centroids[:, j] = np.bincount(
                assignments, weights=x[:, j], minlength=k
            ).astype(np.float32, copy=False)
        centroids /= denom[:, None]
        err = np.zeros(x.shape[0], dtype=np.float32)
        for j in range(x.shape[1]):
            delta = x[:, j] - centroids[assignments, j]
            err += delta * delta
        return centroids, float(err.mean(dtype=np.float64))

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        x = self._to_f32_c(problem["vectors"])
        n, dim = x.shape
        k = int(problem["k"])

        if k <= 1:
            return self._single_cluster(x)
        if k >= n:
            return self._identity_clusters(x)

        work = n * dim * k
        cpu = os.cpu_count() or 1
        if work < 200_000:
            faiss.omp_set_num_threads(1)
        elif work < 2_000_000:
            faiss.omp_set_num_threads(min(2, cpu))
        elif work < 20_000_000:
            faiss.omp_set_num_threads(min(4, cpu))
        else:
            faiss.omp_set_num_threads(min(8, cpu))

        niter = int(kwargs.get("niter", 55 if work < 2_000_000 else 60))
        kmeans = faiss.Kmeans(dim, k, niter=niter, verbose=False)
        kmeans.train(x)

        distances, assignments = kmeans.index.search(x, 1)
        assignments = assignments.reshape(-1)
        counts = np.bincount(assignments, minlength=k)

        if np.all(counts > 0):
            return {
                "centroids": np.asarray(kmeans.centroids, dtype=np.float32).reshape(k, dim),
                "assignments": assignments,
                "quantization_error": float(distances.mean()),
            }

        assignments, counts = self._repair_empty(assignments, distances, k)
        centroids, mse = self._means_and_mse(x, assignments, k, counts)

        return {
            "centroids": centroids,
            "assignments": assignments,
            "quantization_error": mse,
        }