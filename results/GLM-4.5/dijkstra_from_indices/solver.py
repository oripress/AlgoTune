import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any

class Solver:
    def __init__(self):
        # Set default attributes to match reference implementation
        self.directed = True
        self.min_only = True
        
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the shortest path problem using scipy.sparse.csgraph.dijkstra.
        Optimized version with minimal conversions.
        
        :param problem: A dictionary representing the graph (CSR) and source indices.
        :return: A dictionary with key "distances":
                 "distances": A list of shortest path distances from the source nodes.
        """
        try:
            # Create CSR matrix directly from input
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            source_indices = problem["source_indices"]
            
            if not isinstance(source_indices, list) or not source_indices:
                raise ValueError("source_indices missing or empty")
                
            # Run Dijkstra
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=source_indices,
                min_only=self.min_only,
            )
            
            # Convert to required format - fully vectorized version
            mask = np.isinf(dist_matrix)
            result = np.empty_like(dist_matrix, dtype=object)
            result[~mask] = dist_matrix[~mask]
            result[mask] = None
            
            if dist_matrix.ndim == 1:
                dist_matrix_list = [result.tolist()]
            else:
                dist_matrix_list = [row.tolist() for row in result]
            
            return {"distances": dist_matrix_list}
            
        except Exception as e:
            return {"distances": []}