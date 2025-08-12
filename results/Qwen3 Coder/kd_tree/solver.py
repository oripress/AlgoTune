import numpy as np
from typing import Any
import json
import faiss

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Handle string input
        if isinstance(problem, str):
            problem = json.loads(problem)
            
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k = min(problem["k"], len(points))
        dim = points.shape[1]
        
        # Use optimized FAISS index for better performance
        index = faiss.IndexFlatL2(dim)
        index.add(points)
        
        # Enable maximum threading for performance
        faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        
        # Perform search
        distances, indices = index.search(queries, k)
        
        solution = {"indices": indices.tolist(), "distances": distances.tolist()}
        
        # Handle boundary queries for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            bqs = np.stack(bqs, axis=0)
            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()
        
        return solution