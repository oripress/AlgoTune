from typing import Any, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

class Solver:
    def _bruteforce_knn(
        self, points: np.ndarray, queries: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy fallback: exact kNN using per-query argpartition."""
        n_queries = queries.shape[0]
        n_points = points.shape[0]

        idx_res = np.empty((n_queries, k), dtype=np.int64)
        dist_res = np.empty((n_queries, k), dtype=np.float32)

        for i in range(n_queries):
            q = queries[i]
            diff = points - q  # (n_points, dim)
            # Squared L2 distances
            dists = np.einsum("ij,ij->i", diff, diff, optimize=True)

            if k < n_points:
                topk = np.argpartition(dists, k)[:k]
                ord_k = np.argsort(dists[topk], kind="mergesort")
                sel = topk[ord_k]
            else:
                sel = np.argsort(dists)[:k]

            idx_res[i] = sel
            dist_res[i] = dists[sel].astype(np.float32, copy=False)

        return idx_res, dist_res

    @staticmethod
    def _knn_matmul(
        points: np.ndarray, queries: np.ndarray, k: int, p_sq: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact kNN via distance matrix using matmul.
        Returns (indices, distances) with distances as squared L2.
        """
        n_points = points.shape[0]
        n_queries = queries.shape[0]
        if n_queries == 0:
            return (
                np.empty((0, k), dtype=np.int64),
                np.empty((0, k), dtype=np.float32),
            )

        # Precompute squared norms
        if p_sq is None:
            p_sq = np.sum(points * points, axis=1, dtype=np.float32)
        q_sq = np.sum(queries * queries, axis=1, dtype=np.float32).reshape(-1, 1)

        # Distance matrix: ||q||^2 + ||p||^2 - 2 q·p
        G = queries @ points.T  # (nq, np)
        D = q_sq + p_sq[None, :] - 2.0 * G
        # Clamp tiny negatives due to numerical errors
        np.maximum(D, 0.0, out=D)

        if k < n_points:
            idx_part = np.argpartition(D, k - 1, axis=1)[:, :k]
            rows = np.arange(n_queries)[:, None]
            d_part = D[rows, idx_part]
            ord_k = np.argsort(d_part, axis=1, kind="mergesort")
            idx_sorted = np.take_along_axis(idx_part, ord_k, axis=1)
            d_sorted = np.take_along_axis(d_part, ord_k, axis=1)
        else:
            idx_sorted = np.argsort(D, axis=1)[:, :k]
            rows = np.arange(n_queries)[:, None]
            d_sorted = D[rows, idx_sorted]

        return idx_sorted.astype(np.int64, copy=False), d_sorted.astype(np.float32, copy=False)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Extract inputs
        points_list = problem.get("points", [])
        queries_list = problem.get("queries", [])
        k_req = int(problem.get("k", 0))
        n_points = len(points_list)
        n_queries = len(queries_list)

        # Convert to contiguous float32 arrays
        if n_points > 0:
            points = np.ascontiguousarray(np.asarray(points_list, dtype=np.float32))
            dim = points.shape[1]
        else:
            dim = int(problem.get("dim", 0))
            points = np.empty((0, dim), dtype=np.float32)

        if n_queries > 0:
            queries = np.ascontiguousarray(np.asarray(queries_list, dtype=np.float32))
        else:
            queries = np.empty((0, dim), dtype=np.float32)

        # Effective k cannot exceed number of points
        k = min(k_req, n_points)
        # Prepare outputs for k == 0 early
        if k == 0:
            solution = {
                "indices": np.empty((n_queries, 0), dtype=np.int64).tolist(),
                "distances": np.empty((n_queries, 0), dtype=np.float32).tolist(),
            }
            # Boundary handling (ensure expected keys exist when requested)
            if problem.get("distribution") == "hypercube_shell":
                bqs = np.empty((2 * dim, dim), dtype=np.float32)
                bqs[0::2] = 0.0
                bqs[1::2] = 1.0
                solution["boundary_distances"] = np.empty((bqs.shape[0], 0), dtype=np.float32).tolist()
                solution["boundary_indices"] = np.empty((bqs.shape[0], 0), dtype=np.int64).tolist()
            return solution

        # Determine if boundary queries are needed
        want_boundary = problem.get("distribution") == "hypercube_shell"
        if want_boundary:
            # Vectorized construction: 2*dim rows alternating 0-vector and 1-vector
            bqs = np.empty((2 * dim, dim), dtype=np.float32)
            bqs[0::2] = 0.0
            bqs[1::2] = 1.0
            n_bq = bqs.shape[0]
        else:
            bqs = np.empty((0, dim), dtype=np.float32)
            n_bq = 0

        # Initialize boundary arrays to avoid used-before-assignment warnings
        bq_dist = np.empty((0, k), dtype=np.float32)
        bq_idx = np.empty((0, k), dtype=np.int64)

        # Use FAISS if available; otherwise, fallback to exact NumPy
        if faiss is not None and n_points > 0:
            index = faiss.IndexFlatL2(dim)
            index.add(points)

            if want_boundary:
                total_q = n_queries + n_bq
                combined = np.empty((total_q, dim), dtype=np.float32)
                if n_queries > 0:
                    combined[:n_queries] = queries
                combined[n_queries:] = bqs
                d_all, i_all = index.search(combined, k)
                if n_queries > 0:
                    distances = d_all[:n_queries]
                    indices = i_all[:n_queries]
                else:
                    distances = np.empty((0, k), dtype=np.float32)
                    indices = np.empty((0, k), dtype=np.int64)
                bq_dist = d_all[n_queries:]
                bq_idx = i_all[n_queries:]
            else:
                if n_queries > 0:
                    distances, indices = index.search(queries, k)
                else:
                    distances = np.empty((0, k), dtype=np.float32)
                    indices = np.empty((0, k), dtype=np.int64)
        else:
            # No FAISS: choose matmul path; fallback per-query if too large
            # Threshold in number of distances (memory: 4 bytes each)
            mm_thresh = 4_000_000
            if n_queries > 0:
                if n_points * n_queries <= mm_thresh:
                    p_sq = np.sum(points * points, axis=1, dtype=np.float32) if n_points > 0 else np.empty((0,), np.float32)
                    indices, distances = self._knn_matmul(points, queries, k, p_sq=p_sq)
                else:
                    indices, distances = self._bruteforce_knn(points, queries, k)
            else:
                indices = np.empty((0, k), dtype=np.int64)
                distances = np.empty((0, k), dtype=np.float32)

            if want_boundary:
                if n_points * bqs.shape[0] <= mm_thresh:
                    p_sq = np.sum(points * points, axis=1, dtype=np.float32) if n_points > 0 else np.empty((0,), np.float32)
                    bq_idx, bq_dist = self._knn_matmul(points, bqs, k, p_sq=p_sq)
                else:
                    bq_idx, bq_dist = self._bruteforce_knn(points, bqs, k)

        solution = {"indices": indices.tolist(), "distances": distances.tolist()}

        if want_boundary:
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()

        return solution