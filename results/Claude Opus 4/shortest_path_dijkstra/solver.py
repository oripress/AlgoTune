import numpy as np
import scipy.sparse
from typing import Any

class Solver:
    def __init__(self):
        self.method = 'D'  # Dijkstra
        self.directed = False
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the all-pairs shortest path problem.
        """
        try:
            # Create CSR matrix from problem data
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), 
                shape=problem["shape"]
            )
        except Exception:
            return {"distance_matrix": []}
        
        try:
            # Compute shortest paths
            dist_matrix = scipy.sparse.csgraph.shortest_path(
                csgraph=graph_csr, 
                method=self.method, 
                directed=self.directed
            )
        except Exception:
            return {"distance_matrix": []}
        
        # Convert to list format with None for infinity
        dist_matrix_list = [
            [(None if np.isinf(d) else d) for d in row] 
            for row in dist_matrix
        ]
        
        return {"distance_matrix": dist_matrix_list}