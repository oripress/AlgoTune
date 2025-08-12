import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert to numpy arrays efficiently with C-contiguous order
        points = np.array(problem["points"], dtype=np.float32, order='C')
        queries = np.array(problem["queries"], dtype=np.float32, order='C')
        k = problem["k"]
        dim = points.shape[1]
        n_points = len(points)
        n_queries = len(queries)

        # Choose the best algorithm based on problem characteristics
        if n_points < 800 and n_queries < 80:
            # Use scikit-learn for small datasets with optimized settings
            nbrs = NearestNeighbors(n_neighbors=min(k, n_points), algorithm='auto', n_jobs=4)
            nbrs.fit(points)
            distances, indices = nbrs.kneighbors(queries)
            # Convert to squared distances to match faiss output
            distances = distances ** 2
        else:
            # Use faiss for larger datasets with optimized settings
            faiss.omp_set_num_threads(4)
            
            # Use faiss with direct array assignment
            index = faiss.IndexFlatL2(dim)
            index.add(points)
            
            # Use faiss search
            k_search = min(k, n_points)
            distances, indices = index.search(queries, k_search)

        # Convert to lists efficiently
        solution = {
            "indices": indices.tolist(),
            "distances": distances.tolist()
        }

        # Handle hypercube_shell case efficiently
        if problem.get("distribution") == "hypercube_shell":
            # Create boundary queries more efficiently using vectorized operations
            n_boundary = 2 * dim
            bqs = np.zeros((n_boundary, dim), dtype=np.float32, order='C')
            # Vectorized assignment for boundary queries
            dims = np.arange(dim)
            bqs[2*dims, dims] = 0.0
            bqs[2*dims + 1, dims] = 1.0
            
            if n_points < 800 and n_queries < 80:
                # Use scikit-learn for boundary queries too
                bq_distances, bq_indices = nbrs.kneighbors(bqs)
                bq_distances = bq_distances ** 2
            else:
                # Use faiss for boundary queries
                bq_distances, bq_indices = index.search(bqs, k_search)
            
            solution["boundary_distances"] = bq_distances.tolist()
            solution["boundary_indices"] = bq_indices.tolist()

        return solution