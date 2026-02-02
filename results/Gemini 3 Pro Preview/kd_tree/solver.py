import numpy as np
import faiss
from scipy.spatial import cKDTree

class Solver:
    def solve(self, problem, **kwargs):
        points_data = problem["points"]
        queries_data = problem["queries"]
        k = problem["k"]
        
        if isinstance(points_data, np.ndarray):
            points = points_data
        else:
            points = np.array(points_data)
            
        if isinstance(queries_data, np.ndarray):
            queries = queries_data
        else:
            queries = np.array(queries_data)
            
        n_points, dim = points.shape
        n_queries = len(queries)
        k_search = min(k, n_points)
        
        # Heuristics
        # 1. Numpy Matrix Ops for small problems (N*Q small)
        # 2. cKDTree for low dim (d <= 5)
        # 3. Faiss for everything else
        
        use_numpy = (n_points * n_queries < 500000) and (n_points < 10000)
        use_ckdtree = (not use_numpy) and (dim <= 5)
        
        # Helper to perform search
        def search(p, q, k_val):
            if use_numpy:
                # Ensure float64 for precision if possible, else float32
                dtype = np.float64 if (p.dtype == np.float64 or q.dtype == np.float64) else np.float32
                p_ = p.astype(dtype, copy=False)
                q_ = q.astype(dtype, copy=False)
                
                # (Q, D) @ (D, N) -> (Q, N)
                dot = np.dot(q_, p_.T)
                p_sq = np.sum(p_**2, axis=1)
                q_sq = np.sum(q_**2, axis=1)
                
                dists_sq = q_sq[:, np.newaxis] + p_sq[np.newaxis, :] - 2*dot
                dists_sq[dists_sq < 0] = 0
                
                if k_val < len(p):
                    idx = np.argpartition(dists_sq, k_val-1, axis=1)[:, :k_val]
                    top_dists = np.take_along_axis(dists_sq, idx, axis=1)
                    sort_idx = np.argsort(top_dists, axis=1)
                    indices = np.take_along_axis(idx, sort_idx, axis=1)
                    distances = np.take_along_axis(top_dists, sort_idx, axis=1)
                else:
                    indices = np.argsort(dists_sq, axis=1)
                    distances = np.take_along_axis(dists_sq, indices, axis=1)
                return distances, indices
                
            elif use_ckdtree:
                # tree is defined in outer scope if use_ckdtree is True
                dists, indices = tree.query(q, k=k_val, workers=-1)
                if k_val == 1:
                    dists = dists[:, np.newaxis]
                    indices = indices[:, np.newaxis]
                return dists**2, indices
                
            else:
                # index is defined in outer scope
                # Faiss requires float32
                q_f = np.ascontiguousarray(q, dtype=np.float32)
                return index.search(q_f, k_val)

        # Initialize structures
        tree = None
        index = None
        
        if use_ckdtree:
            tree = cKDTree(points, leafsize=16)
        elif not use_numpy:
            p_f = np.ascontiguousarray(points, dtype=np.float32)
            index = faiss.IndexFlatL2(dim)
            index.add(p_f)
            
        distances, indices = search(points, queries, k_search)
        
        solution = {"indices": indices.tolist(), "distances": distances.tolist()}
        
        if problem.get("distribution") == "hypercube_shell":
            bqs = np.empty((2 * dim, dim), dtype=points.dtype)
            bqs[0::2] = 0
            bqs[1::2] = 1
            
            bq_dist, bq_idx = search(points, bqs, k_search)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()
                
        return solution