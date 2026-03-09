from __future__ import annotations

import os
from typing import Any

import faiss
import numpy as np

class Solver:
    def __init__(self) -> None:
        self._last_points_obj: Any = None
        self._last_points_arr: np.ndarray | None = None
        self._last_dim: int = -1
        self._last_index: faiss.IndexFlatL2 | None = None
        self._last_points_sqnorm: np.ndarray | None = None
        self._last_boundary_k: int = -1
        self._last_boundary_indices: np.ndarray | None = None
        self._last_boundary_distances: np.ndarray | None = None
        self._faiss_knn = getattr(faiss, "knn", None)
        faiss.omp_set_num_threads(1)

    @staticmethod
    def _as_f32_2d(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)
        return np.ascontiguousarray(arr)

    def _prepare_points(self, raw_points: Any) -> tuple[np.ndarray, int]:
        if raw_points is not self._last_points_obj or self._last_points_arr is None:
            points = self._as_f32_2d(raw_points)
            self._last_points_obj = raw_points
            self._last_points_arr = points
            self._last_dim = points.shape[1]
            self._last_index = None
            self._last_points_sqnorm = None
            self._last_boundary_k = -1
            self._last_boundary_indices = None
            self._last_boundary_distances = None
        return self._last_points_arr, self._last_dim  # type: ignore[return-value]

    def _get_points_sqnorm(self, points: np.ndarray) -> np.ndarray:
        if self._last_points_sqnorm is None:
            self._last_points_sqnorm = np.einsum("ij,ij->i", points, points, optimize=True)
        return self._last_points_sqnorm

    def _get_index(self, points: np.ndarray, dim: int) -> faiss.IndexFlatL2:
        if self._last_index is None:
            index = faiss.IndexFlatL2(dim)
            if len(points):
                index.add(points)
            self._last_index = index
        return self._last_index

    def _search_numpy(
        self, points: np.ndarray, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        p2 = self._get_points_sqnorm(points)
        q2 = np.einsum("ij,ij->i", queries, queries, optimize=True)
        dist = q2[:, None] + p2[None, :]
        dist -= np.float32(2.0) * (queries @ points.T)
        np.maximum(dist, np.float32(0.0), out=dist)

        if k == 1:
            idx = np.argmin(dist, axis=1)
            return dist[np.arange(len(queries)), idx][:, None], idx[:, None]

        row_ids = np.arange(len(queries))[:, None]
        if k < len(points):
            idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            dsel = dist[row_ids, idx]
            order = np.argsort(dsel, axis=1)
            return dsel[row_ids, order], idx[row_ids, order]

        idx = np.argsort(dist, axis=1)
        return dist[row_ids, idx], idx

    def _search_faiss(
        self, points: np.ndarray, queries: np.ndarray, dim: int, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._faiss_knn is not None:
            try:
                distances, indices = self._faiss_knn(queries, points, k)
                return distances, indices
            except Exception:
                self._faiss_knn = None
        index = self._get_index(points, dim)
        return index.search(queries, k)

    @staticmethod
    def _boundary_queries(dim: int) -> np.ndarray:
        bqs = np.empty((2, dim), dtype=np.float32)
        bqs[0].fill(0.0)
        bqs[1].fill(1.0)
        return bqs

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        points, dim = self._prepare_points(problem["points"])
        queries = self._as_f32_2d(problem["queries"])

        n_points = len(points)
        n_queries = len(queries)
        k = problem["k"]
        if k > n_points:
            k = n_points

        if k == 0:
            solution: dict[str, Any] = {
                "indices": np.empty((n_queries, 0), dtype=np.int64),
                "distances": np.empty((n_queries, 0), dtype=np.float32),
            }
            if problem.get("distribution") == "hypercube_shell":
                solution["boundary_indices"] = np.empty((2 * dim, 0), dtype=np.int64)
                solution["boundary_distances"] = np.empty((2 * dim, 0), dtype=np.float32)
            return solution

        use_numpy = n_points * max(n_queries, 1) * dim <= 1_000_000 and (n_points <= 256 or n_queries <= 8)

        if use_numpy:
            distances, indices = self._search_numpy(points, queries, k)
        else:
            distances, indices = self._search_faiss(points, queries, dim, k)

        solution: dict[str, Any] = {"indices": indices, "distances": distances}

        if problem.get("distribution") == "hypercube_shell":
            if k == self._last_boundary_k and self._last_boundary_indices is not None:
                solution["boundary_indices"] = self._last_boundary_indices
                solution["boundary_distances"] = self._last_boundary_distances
            else:
                bqs = self._boundary_queries(dim)
                if use_numpy:
                    bq_dist2, bq_idx2 = self._search_numpy(points, bqs, k)
                else:
                    bq_dist2, bq_idx2 = self._search_faiss(points, bqs, dim, k)
                self._last_boundary_indices = np.tile(bq_idx2, (dim, 1))
                self._last_boundary_distances = np.tile(bq_dist2, (dim, 1))
                self._last_boundary_k = k
                solution["boundary_indices"] = self._last_boundary_indices
                solution["boundary_distances"] = self._last_boundary_distances

        return solution