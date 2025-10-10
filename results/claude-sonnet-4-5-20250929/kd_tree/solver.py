import numpy as np
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = problem["points"]
        queries = problem["queries"]
        k = min(problem["k"], len(points))
        
        # Fast conversion to float32 numpy arrays
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        elif points.dtype != np.float32:
            points = points.astype(np.float32)
            
        if not isinstance(queries, np.ndarray):
            queries = np.array(queries, dtype=np.float32)
        elif queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        
        dim = points.shape[1]
        
        # Use IndexFlatL2 directly
        index = faiss.IndexFlatL2(dim)
        index.add(points)
        
        distances, indices = index.search(queries, k)
        
        solution = {"indices": indices.tolist(), "distances": distances.tolist()}
        
        # Handle hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            bqs_list = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs_list.append(q0)
                bqs_list.append(q1)
            bqs = np.array(bqs_list, dtype=np.float32)
            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()
        
        return solution