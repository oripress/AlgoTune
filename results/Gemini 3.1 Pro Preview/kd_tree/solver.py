import faiss
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k = min(problem["k"], len(points))
        dim = points.shape[1]

        if dim <= 10:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances, indices = tree.query(queries, k=k, workers=-1)
            distances = distances ** 2
            if k == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)
            distances = distances.astype(np.float32)
            indices = indices.astype(np.int64)
        else:
            distances, indices = faiss.knn(queries, points, k)

        solution = {
            "indices": indices,
            "distances": distances
        }

        if problem.get("distribution") == "hypercube_shell":
            bqs = np.empty((2 * dim, dim), dtype=np.float32)
            bqs[0::2] = 0.0
            bqs[1::2] = 1.0
            
            if dim <= 10:
                bq_dist, bq_idx = tree.query(bqs, k=k)
                bq_dist = (bq_dist ** 2).astype(np.float32)
                if k == 1:
                    bq_dist = bq_dist.reshape(-1, 1)
                    bq_idx = bq_idx.reshape(-1, 1)
            else:
                bq_dist, bq_idx = faiss.knn(bqs, points, k)
                
            solution["boundary_distances"] = bq_dist
            solution["boundary_indices"] = bq_idx

        return solution
        return solution