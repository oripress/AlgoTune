import numpy as np
from typing import Any
from scipy.spatial import KDTree

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k = problem["k"]
        dim = points.shape[1]
        n_points = len(points)
        
        # Build KD-tree using scipy
        kdtree = KDTree(points)
        
        # Find k nearest neighbors for each query
        k = min(k, n_points)
        
        # Query all at once for efficiency
        distances, indices = kdtree.query(queries, k=k)
        
        # Square the distances since scipy returns Euclidean distance but we need squared
        distances = distances ** 2
        
        # Convert to the expected format
        if k == 1:
            # When k=1, query returns single values, not arrays
            distances = [[float(d)] for d in distances]
            indices = [[int(i)] for i in indices]
        else:
            distances = distances.tolist()
            indices = indices.tolist()
        
        solution = {"indices": indices, "distances": distances}
        
        # Handle boundary queries for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            
            bqs = np.array(bqs)
            bq_distances, bq_indices = kdtree.query(bqs, k=k)
            
            if k == 1:
                bq_distances = [[float(d)] for d in bq_distances]
                bq_indices = [[int(i)] for i in bq_indices]
            else:
                bq_distances = bq_distances.tolist()
                bq_indices = bq_indices.tolist()
            
            solution["boundary_distances"] = bq_distances
            solution["boundary_indices"] = bq_indices
        
        return solution