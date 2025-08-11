import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Optimized Dijkstra using scipy with minimal overhead.
        """
        # Create CSR matrix directly without copying
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
            dtype=np.float64
        )
        
        # Compute shortest paths
        dist_matrix = scipy.sparse.csgraph.dijkstra(
            csgraph=graph_csr,
            directed=False,
            indices=problem["source_indices"],
            min_only=False,
            return_predecessors=False
        )
        
        # Efficient conversion to required format
        if dist_matrix.ndim == 1:
            dist_matrix = dist_matrix.reshape(1, -1)
        
        # Convert infinity to None efficiently
        result = []
        for row in dist_matrix:
            converted_row = []
            for val in row:
                if np.isinf(val):
                    converted_row.append(None)
                else:
                    converted_row.append(float(val))
            result.append(converted_row)
        
        return {"distances": result}