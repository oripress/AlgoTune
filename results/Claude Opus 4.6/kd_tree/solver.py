import numpy as np
from typing import Any
import faiss
import os
from scipy.spatial import cKDTree

# Set threads for FAISS
n_threads = os.cpu_count() or 1
faiss.omp_set_num_threads(n_threads)

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        points_raw = problem["points"]
        queries_raw = problem["queries"]
        k = problem["k"]
        
        # Convert to numpy arrays efficiently - avoid copy if already correct type
        if isinstance(points_raw, np.ndarray) and points_raw.dtype == np.float32:
            points = np.ascontiguousarray(points_raw)
        elif isinstance(points_raw, np.ndarray):
            points = np.ascontiguousarray(points_raw, dtype=np.float32)
        else:
            points = np.array(points_raw, dtype=np.float32)
        
        if isinstance(queries_raw, np.ndarray) and queries_raw.dtype == np.float32:
            queries = np.ascontiguousarray(queries_raw)
        elif isinstance(queries_raw, np.ndarray):
            queries = np.ascontiguousarray(queries_raw, dtype=np.float32)
        else:
            queries = np.array(queries_raw, dtype=np.float32)
        
        n_points = points.shape[0]
        n_queries = queries.shape[0]
        dim = points.shape[1]
        k = min(k, n_points)
        
        if n_points <= 0 or k <= 0:
            return {"indices": [[] for _ in range(n_queries)], 
                    "distances": [[] for _ in range(n_queries)]}
        
        is_shell = problem.get("distribution") == "hypercube_shell"
        
        # For low dimensions, cKDTree is O(n log n) build + O(log n) query
        # vs FAISS brute force O(n*q*d)
        use_kdtree = dim <= 10 and n_points >= 200 and n_points * n_queries > 50000
        
        if use_kdtree:
            pts64 = points.astype(np.float64)
            qrs64 = queries.astype(np.float64)
            tree = cKDTree(pts64, leafsize=16, compact_nodes=False, balanced_tree=False)
            dists, indices = tree.query(qrs64, k=k, workers=-1)
            if k == 1:
                dists = dists.reshape(n_queries, 1)
                indices = indices.reshape(n_queries, 1)
            distances = np.empty_like(dists, dtype=np.float32)
            np.multiply(dists, dists, out=distances)
            indices = np.asarray(indices, dtype=np.int64)
            
            # For shell boundary queries, use FAISS
            if is_shell:
                index = faiss.IndexFlatL2(dim)
                index.add(points)
        else:
            index = faiss.IndexFlatL2(dim)
            index.add(points)
            distances, indices = index.search(queries, k)
        
        solution = {"indices": indices, "distances": distances}
        
        if is_shell:
            bqs = np.zeros((2 * dim, dim), dtype=np.float32)
            for d in range(dim):
                bqs[2*d + 1] = 1.0
            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist
            solution["boundary_indices"] = bq_idx
        
        return solution