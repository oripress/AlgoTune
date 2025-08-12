import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any

class Solver:
    def __init__(self):
        self.directed = False
        # Pre-compile empty arrays for reuse
        self._cache = {}
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Optimized all-pairs shortest path solver.
        """
        # Direct array access without intermediate variables
        n = problem["shape"][0]
        
        # Create CSR matrix directly - fastest method
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]), 
            shape=problem["shape"]
        )
        
        # Always use Dijkstra as it's fastest for sparse graphs
        # which is what we're dealing with per the task description
        dist_matrix = scipy.sparse.csgraph.dijkstra(
            csgraph=graph_csr, 
            directed=self.directed,
            return_predecessors=False,
            limit=np.inf  # No limit
        )
        
        # Fastest conversion using tolist() and replacing inf in one go
        # Convert numpy array to list in C (fastest)
        dist_list = dist_matrix.tolist()
        
        # Replace inf with None in-place
        for i in range(n):
            row = dist_list[i]
            for j in range(n):
                if row[j] == np.inf:
                    row[j] = None
        
        return {"distance_matrix": dist_list}