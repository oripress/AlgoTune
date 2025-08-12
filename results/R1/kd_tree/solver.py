import numpy as np
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Convert inputs to float32 for FAISS optimization
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k_val = problem["k"]
        dim = points.shape[1]
        
        # Set FAISS to use all available threads
        faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        
        # Create optimized FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(points)
        
        # Ensure k is not larger than the number of points
        k_val = min(k_val, len(points))
        
        # Perform main search with maximum parallelism
        distances, indices = index.search(queries, k_val)
        
        solution = {
            "indices": indices.tolist(),
            "distances": distances.tolist()
        }
        
        # Boundary queries for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            bqs = np.stack(bqs, axis=0)
            
            # Compute boundary queries with maximum parallelism
            bq_dist, bq_idx = index.search(bqs, k_val)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()
            
        return solution