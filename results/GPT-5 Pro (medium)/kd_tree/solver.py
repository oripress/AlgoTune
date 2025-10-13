from typing import Any, Dict
import numpy as np

try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _HAVE_FAISS = False

class Solver:
    def _search_numpy(self, points: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # points: (n, d), queries: (m, d)
        n = points.shape[0]
        m = queries.shape[0]
        if n == 0 or k == 0:
            # Return empty results with the right shapes
            return np.empty((m, 0), dtype=np.float32), np.empty((m, 0), dtype=np.int64)

        # Compute squared L2 distances via expansion: ||x||^2 + ||q||^2 - 2 q.x
        x2 = np.einsum("ij,ij->i", points, points)  # (n,)
        q2 = np.einsum("ij,ij->i", queries, queries)  # (m,)
        cross = queries @ points.T  # (m, n)
        dists = q2[:, None] + x2[None, :] - 2.0 * cross
        # Numerical stability
        np.maximum(dists, 0.0, out=dists)

        if k < n:
            idx_part = np.argpartition(dists, k - 1, axis=1)[:, :k]
        else:
            idx_part = np.broadcast_to(np.arange(n, dtype=np.int64), (m, n)).copy()

        part_dists = np.take_along_axis(dists, idx_part, axis=1)
        order = np.argsort(part_dists, axis=1, kind="stable")
        sorted_idx = np.take_along_axis(idx_part, order, axis=1).astype(np.int64, copy=False)
        sorted_dists = np.take_along_axis(part_dists, order, axis=1).astype(np.float32, copy=False)
        return sorted_dists, sorted_idx

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float32)
        queries = np.asarray(problem["queries"], dtype=np.float32)
        k_req = int(problem["k"])
        n_points = points.shape[0]
        dim = points.shape[1] if n_points > 0 else (queries.shape[1] if queries.size else 0)
        k = min(k_req, n_points)

        # Ensure contiguous memory for FAISS
        points = np.ascontiguousarray(points)
        queries = np.ascontiguousarray(queries)

        if _HAVE_FAISS and dim > 0 and n_points > 0:
            index = faiss.IndexFlatL2(dim)
            index.add(points)
            distances, indices = index.search(queries, k)
        else:
            distances, indices = self._search_numpy(points, queries, k)

        solution: Dict[str, Any] = {
            "indices": indices.tolist(),
            "distances": distances.tolist(),
        }

        # Boundary queries for "hypercube_shell" distribution (replicate reference behavior)
        if problem.get("distribution") == "hypercube_shell" and dim > 0:
            bqs_list = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs_list.extend([q0, q1])
            bqs = np.stack(bqs_list, axis=0)

            if _HAVE_FAISS and n_points > 0:
                bq_dist, bq_idx = index.search(bqs, k)
            else:
                bq_dist, bq_idx = self._search_numpy(points, bqs, k)

            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()

        return solution