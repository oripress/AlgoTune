import numpy as np
from typing import Any
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the all-pairs shortest path problem using scipy with optimized Dijkstra.
        :param problem: A dictionary representing the graph in CSR components.
        :return: A dictionary with key "distance_matrix":
                 "distance_matrix": The matrix of shortest path distances (list of lists).
                                     None indicates no path.
        """
        # Extract CSR components with minimal copying
        data = np.asarray(problem["data"], dtype=np.float64)
        indices = np.asarray(problem["indices"], dtype=np.int32)
        indptr = np.asarray(problem["indptr"], dtype=np.int32)
        shape = problem["shape"]
        
        # Create CSR matrix directly without intermediate conversions
        graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)
        
        # Use Dijkstra algorithm which is often better for sparse graphs
        dist_matrix = scipy.sparse.csgraph.shortest_path(
            csgraph=graph_csr, method='D', directed=False
        )
        
        # Most efficient conversion to list of lists with None for infinity
        # Use vectorized numpy operations
        inf_mask = np.isinf(dist_matrix)
        dist_matrix[inf_mask] = np.nan  # Use NaN as temporary placeholder
        dist_matrix_list = dist_matrix.tolist()
        
        # Replace NaN with None in the nested list structure
        for i in range(len(dist_matrix_list)):
            for j in range(len(dist_matrix_list[i])):
                if dist_matrix_list[i][j] != dist_matrix_list[i][j]:  # Check for NaN
                    dist_matrix_list[i][j] = None
        
        return {"distance_matrix": dist_matrix_list}