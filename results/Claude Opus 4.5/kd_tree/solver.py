import numpy as np
import faiss
from scipy.spatial import cKDTree
from typing import Any
import os

class Solver:
    def __init__(self):
        n_threads = os.cpu_count() or 4
        faiss.omp_set_num_threads(n_threads)
        self.n_threads = n_threads
    
    def solve(self, problem, **kwargs) -> Any:
        points = np.asarray(problem["points"], dtype=np.float64)
        queries = np.asarray(problem["queries"], dtype=np.float64)
        k = min(problem["k"], len(points))
        dim = points.shape[1]
        n_points = len(points)
        
        # Use cKDTree for low dimensions (more efficient)
        if dim <= 10:
            tree = cKDTree(points, leafsize=32, balanced_tree=False)
            dists, indices = tree.query(queries, k=k, workers=-1)
            # Square distances to match FAISS output
            if k == 1:
                indices = indices.reshape(-1, 1)
                dists = dists.reshape(-1, 1)
            distances = np.square(dists)
        else:
            # Use FAISS for high dimensions
            points32 = np.ascontiguousarray(points, dtype=np.float32)
            queries32 = np.ascontiguousarray(queries, dtype=np.float32)
            index = faiss.IndexFlatL2(dim)
            index.add(points32)
            distances, indices = index.search(queries32, k)
        
        solution = {"indices": indices.tolist(), "distances": distances.tolist()}
        
        # Handle hypercube_shell boundary queries
        if problem.get("distribution") == "hypercube_shell":
            bqs = np.zeros((2 * dim, dim), dtype=np.float64)
            for d_idx in range(dim):
                bqs[2 * d_idx + 1, :] = 1.0
            if dim <= 20:
                bq_dist, bq_idx = tree.query(bqs, k=k, workers=self.n_threads)
                if k == 1:
                    bq_idx = bq_idx.reshape(-1, 1)
                    bq_dist = bq_dist.reshape(-1, 1)
                bq_dist = np.square(bq_dist)
            else:
                bqs32 = np.ascontiguousarray(bqs, dtype=np.float32)
                bq_dist, bq_idx = index.search(bqs32, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()
        
        return solution