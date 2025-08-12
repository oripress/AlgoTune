import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k = problem["k"]
        n_points = len(points)
        dim = points.shape[1]
        
        # Use IndexFlatL2 for exact nearest neighbor search
        index = faiss.IndexFlatL2(dim)
        index.add(points)
        
        # Ensure k doesn't exceed number of points
        k = min(k, n_points)
        
        # Search for k nearest neighbors
        distances, indices = index.search(queries, k)
        
        solution = {
            "indices": indices.tolist(),
            "distances": distances.tolist()
        }
        
        # Handle boundary queries for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            bqs = np.array(bqs, dtype=np.float32)
            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()
        
        return solution