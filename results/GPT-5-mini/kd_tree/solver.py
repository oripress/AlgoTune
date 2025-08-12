from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        k-NN solver that returns squared L2 distances.

        Strategy:
          - Try faiss.IndexFlatL2 + IndexIDMap if available (fast).
          - Fallback to sklearn.neighbors.KDTree if available.
          - Final fallback: batched NumPy brute-force using (a-b)^2 = a^2 + b^2 - 2ab.

        Returns:
            {"indices": [[...], ...], "distances": [[...], ...]}
        Optionally includes "boundary_indices" and "boundary_distances" when
        problem.get("distribution") == "hypercube_shell".
        """
        pts_in = problem.get("points", [])
        q_in = problem.get("queries", [])
        k_req = int(problem.get("k", 0))
        dim_given = problem.get("dim", None)
        distribution = problem.get("distribution", None)

        def to_2d(arr, dim_hint=None, dtype=np.float64):
            a = np.asarray(arr, dtype=dtype)
            if a.size == 0:
                if dim_hint is None:
                    return np.empty((0, 0), dtype=dtype)
                return np.empty((0, dim_hint), dtype=dtype)
            if a.ndim == 1:
                return a.reshape(1, -1)
            if a.ndim == 2:
                return a
            return a.reshape(a.shape[0], -1)

        # Infer dimension if not provided
        if dim_given is not None:
            dim = int(dim_given)
        else:
            dim = 0
            if isinstance(pts_in, (list, tuple)) and len(pts_in) > 0:
                dim = int(np.asarray(pts_in[0]).size)
            elif isinstance(q_in, (list, tuple)) and len(q_in) > 0:
                dim = int(np.asarray(q_in[0]).size)

        points = to_2d(pts_in, dim_hint=dim, dtype=np.float64)
        queries = to_2d(q_in, dim_hint=dim, dtype=np.float64)

        n_points = int(points.shape[0])
        n_queries = int(queries.shape[0])
        k = min(k_req, n_points)

        # Trivial cases
        if n_queries == 0:
            solution = {
                "indices": np.empty((0, k), dtype=int).tolist(),
                "distances": np.empty((0, k), dtype=float).tolist(),
            }
            if distribution == "hypercube_shell":
                solution["boundary_distances"] = []
                solution["boundary_indices"] = []
            return solution

        if n_points == 0 or k == 0:
            solution = {
                "indices": np.empty((n_queries, 0), dtype=int).tolist(),
                "distances": np.empty((n_queries, 0), dtype=float).tolist(),
            }
            if distribution == "hypercube_shell":
                solution["boundary_distances"] = []
                solution["boundary_indices"] = []
            return solution

        # Helper to compute k-NN from squared-distance matrix
        def knn_from_dmat(dmat: np.ndarray, kk: int):
            m, npnts = dmat.shape
            if kk <= 0:
                return np.empty((m, 0), dtype=int), np.empty((m, 0), dtype=float)
            if kk >= npnts:
                idx_sorted = np.argsort(dmat, axis=1)
                d_sorted = np.take_along_axis(dmat, idx_sorted, axis=1)
                return idx_sorted, d_sorted
            idx_uns = np.argpartition(dmat, kth=kk - 1, axis=1)[:, :kk]
            sel = np.take_along_axis(dmat, idx_uns, axis=1)
            order = np.argsort(sel, axis=1)
            idx_sorted = np.take_along_axis(idx_uns, order, axis=1)
            d_sorted = np.take_along_axis(sel, order, axis=1)
            return idx_sorted, d_sorted

        # 1) Try faiss (squared L2 distances)
        try:
            import faiss  # type: ignore

            pts32 = np.ascontiguousarray(points.astype(np.float32))
            q32 = np.ascontiguousarray(queries.astype(np.float32))
            dim_f = pts32.shape[1] if pts32.size else int(dim)
            index = faiss.IndexFlatL2(dim_f)
            index = faiss.IndexIDMap(index)
            ids = np.arange(n_points, dtype=np.int64)
            index.add_with_ids(pts32, ids)
            distances, indices = index.search(q32, k)
            solution = {"indices": indices.tolist(), "distances": distances.tolist()}

            if distribution == "hypercube_shell":
                bqs = []
                for d in range(dim):
                    q0 = np.zeros(dim, dtype=np.float32)
                    q1 = np.ones(dim, dtype=np.float32)
                    bqs.extend([q0, q1])
                if bqs:
                    bqs = np.stack(bqs, axis=0)
                    bq_dist, bq_idx = index.search(bqs.astype(np.float32), k)
                    solution["boundary_distances"] = bq_dist.tolist()
                    solution["boundary_indices"] = bq_idx.tolist()
                else:
                    solution["boundary_distances"] = []
                    solution["boundary_indices"] = []
            return solution
        except Exception:
            # faiss not available or failed; fall back
            pass

        # 2) Try scipy.spatial.cKDTree (fast C implementation), then sklearn KDTree
        try:
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(points.astype(np.float64))
            dist, idx = tree.query(queries.astype(np.float64), k=k)
            # Normalize shapes to (n_queries, k)
            if np.isscalar(idx) or idx.ndim == 1:
                if np.isscalar(idx):
                    idx = np.array([[int(idx)]])
                    dist = np.array([[float(dist)]])
                else:
                    idx = idx.reshape(-1, 1)
                    dist = dist.reshape(-1, 1)
            solution = {"indices": idx.tolist(), "distances": (dist ** 2).tolist()}

            if distribution == "hypercube_shell":
                bqs = []
                for d in range(dim):
                    bqs.append(np.zeros(dim, dtype=np.float64))
                    bqs.append(np.ones(dim, dtype=np.float64))
                if bqs:
                    bqs = np.stack(bqs, axis=0)
                    bq_dist, bq_idx = tree.query(bqs, k=k)
                    if np.isscalar(bq_idx) or bq_idx.ndim == 1:
                        if np.isscalar(bq_idx):
                            bq_idx = np.array([[int(bq_idx)]])
                            bq_dist = np.array([[float(bq_dist)]])
                        else:
                            bq_idx = bq_idx.reshape(-1, 1)
                            bq_dist = bq_dist.reshape(-1, 1)
                    solution["boundary_distances"] = (bq_dist ** 2).tolist()
                    solution["boundary_indices"] = bq_idx.tolist()
                else:
                    solution["boundary_distances"] = []
                    solution["boundary_indices"] = []
            return solution
        except Exception:
            # scipy not available or failed; fall back to sklearn KDTree
            pass

        # 3) Try sklearn KDTree (returns Euclidean distances -> square them)
        try:
            from sklearn.neighbors import KDTree  # type: ignore

            tree = KDTree(points.astype(np.float64))
            dist, idx = tree.query(queries.astype(np.float64), k=k)
            # Ensure shapes are (n_queries, k)
            if idx.ndim == 1:
                idx = idx.reshape(-1, 1)
                dist = dist.reshape(-1, 1)
            solution = {"indices": idx.tolist(), "distances": (dist ** 2).tolist()}

            if distribution == "hypercube_shell":
                bqs = []
                for d in range(dim):
                    bqs.append(np.zeros(dim, dtype=np.float64))
                    bqs.append(np.ones(dim, dtype=np.float64))
                if bqs:
                    bqs = np.stack(bqs, axis=0)
                    bq_dist, bq_idx = tree.query(bqs, k=k)
                    if bq_idx.ndim == 1:
                        bq_idx = bq_idx.reshape(-1, 1)
                        bq_dist = bq_dist.reshape(-1, 1)
                    solution["boundary_distances"] = (bq_dist ** 2).tolist()
                    solution["boundary_indices"] = bq_idx.tolist()
                else:
                    solution["boundary_distances"] = []
                    solution["boundary_indices"] = []
            return solution
        except Exception:
            # sklearn not available or failed; fall back to numpy brute-force
            pass
            pass

        # 3) NumPy batched brute-force (squared distances)
        pts = points.astype(np.float64)
        qrs = queries.astype(np.float64)

        # Use identity: (a-b)^2 = a^2 + b^2 - 2ab to avoid large temporary tensors when possible
        pts_sq = np.sum(pts * pts, axis=1)  # (n_points,)

        max_cells = 10_000_000  # heuristic cap on n_points * n_queries

        if n_queries * n_points <= max_cells:
            q_sq = np.sum(qrs * qrs, axis=1)  # (n_queries,)
            dmat = q_sq[:, None] + pts_sq[None, :] - 2.0 * (qrs @ pts.T)
            # numerical safety
            np.maximum(dmat, 0.0, out=dmat)
            idx_sorted, d_sorted = knn_from_dmat(dmat, k)
            solution = {"indices": idx_sorted.tolist(), "distances": d_sorted.tolist()}
        else:
            batch_q = max(1, int(max_cells // n_points))
            parts_idx = []
            parts_d = []
            for s in range(0, n_queries, batch_q):
                e = min(n_queries, s + batch_q)
                qb = qrs[s:e]
                qb_sq = np.sum(qb * qb, axis=1)
                dmat = qb_sq[:, None] + pts_sq[None, :] - 2.0 * (qb @ pts.T)
                np.maximum(dmat, 0.0, out=dmat)
                idx_sorted, d_sorted = knn_from_dmat(dmat, k)
                parts_idx.append(idx_sorted)
                parts_d.append(d_sorted)
            if parts_idx:
                indices_all = np.vstack(parts_idx)
                distances_all = np.vstack(parts_d)
            else:
                indices_all = np.empty((0, k), dtype=int)
                distances_all = np.empty((0, k), dtype=float)
            solution = {"indices": indices_all.tolist(), "distances": distances_all.tolist()}

        # Boundary handling for hypercube_shell
        if distribution == "hypercube_shell":
            bqs = []
            for d in range(dim):
                bqs.append(np.zeros(dim, dtype=np.float64))
                bqs.append(np.ones(dim, dtype=np.float64))
            if bqs:
                bq = np.stack(bqs, axis=0)
                bq_sq = np.sum(bq * bq, axis=1)
                bd = bq_sq[:, None] + pts_sq[None, :] - 2.0 * (bq @ pts.T)
                np.maximum(bd, 0.0, out=bd)
                bidx, bd_sorted = knn_from_dmat(bd, k)
                solution["boundary_indices"] = bidx.tolist()
                solution["boundary_distances"] = bd_sorted.tolist()
            else:
                solution["boundary_distances"] = []
                solution["boundary_indices"] = []

        return solution