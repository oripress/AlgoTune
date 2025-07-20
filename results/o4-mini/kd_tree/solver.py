import numpy as np
from typing import Any, Dict

# Try to import Faiss for fast search
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        points = problem["points"]
        queries = problem["queries"]
        n_points = len(points)
        dim = len(points[0]) if n_points > 0 else 0
        k = min(problem.get("k", 1), n_points)
        # Prepare arrays
        if n_points > 0 and dim > 0:
            pts = np.ascontiguousarray(points, dtype=np.float32).reshape(n_points, dim)
            qrs = np.ascontiguousarray(queries, dtype=np.float32).reshape(len(queries), dim)
        else:
            pts = np.zeros((0, dim), dtype=np.float32)
            qrs = np.zeros((len(queries), dim), dtype=np.float32)

        if _HAS_FAISS and n_points > 0 and dim > 0:
            # maximize threads
            try:
                faiss.omp_set_num_threads(faiss.omp_get_max_threads())
            except Exception:
                pass
            # choose index: HNSW for large sets, Flat otherwise
            if n_points > 20000:
                index = faiss.IndexHNSWFlat(dim, 32)
                index.hnsw.efConstruction = 40
                index.add(pts)
                index.hnsw.efSearch = max(64, k * 8)
            else:
                index = faiss.IndexFlatL2(dim)
                index.add(pts)
            # search queries
            D, I = index.search(qrs, k)
            indices = I
            distances = D.astype(np.float64)
        else:
            # fallback brute force (double precision)
            pts2 = np.asarray(points, dtype=np.float64)
            qrs2 = np.asarray(queries, dtype=np.float64)
            n_queries = len(queries)
            indices = np.empty((n_queries, k), dtype=np.int64)
            distances = np.empty((n_queries, k), dtype=np.float64)
            chunk = 512
            for i in range(0, n_queries, chunk):
                j = min(i + chunk, n_queries)
                sub = qrs2[i:j]  # (chunk, dim)
                # pairwise distances: (sub[:,None,:] - pts2[None,:,:])^2 sum over axis 2
                diff = sub[:, None, :] - pts2[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                # k smallest indices
                idxp = np.argpartition(dist2, k - 1, axis=1)[:, :k]
                for ii in range(j - i):
                    row = idxp[ii]
                    ds = dist2[ii, row]
                    order = np.argsort(ds)
                    indices[i + ii] = row[order]
                    distances[i + ii] = ds[order]

        solution: Dict[str, Any] = {
            "indices": indices.tolist(),
            "distances": distances.tolist(),
        }

        # Boundary queries for hypercube_shell
        if problem.get("distribution") == "hypercube_shell":
            bq_n = dim * 2
            if _HAS_FAISS and n_points > 0 and dim > 0:
                bqs = np.zeros((bq_n, dim), dtype=np.float32)
                bqs[1::2] = np.eye(dim, dtype=np.float32)
                D2, I2 = index.search(bqs, k)
                solution["boundary_indices"] = I2.tolist()
                solution["boundary_distances"] = D2.astype(np.float64).tolist()
            else:
                # brute boundary
                bqs = np.zeros((bq_n, dim), dtype=np.float64)
                bqs[1::2] = np.eye(dim, dtype=np.float64)
                pts2 = np.asarray(points, dtype=np.float64)
                diffb = bqs[:, None, :] - pts2[None, :, :]
                dist2b = np.sum(diffb * diffb, axis=2)
                idxp = np.argpartition(dist2b, k - 1, axis=1)[:, :k]
                b_idx = []
                b_dist = []
                for ii in range(bq_n):
                    row = idxp[ii]
                    ds = dist2b[ii, row]
                    order = np.argsort(ds)
                    b_idx.append([int(x) for x in row[order]])
                    b_dist.append([float(x) for x in ds[order]])
                solution["boundary_indices"] = b_idx
                solution["boundary_distances"] = b_dist

        return solution