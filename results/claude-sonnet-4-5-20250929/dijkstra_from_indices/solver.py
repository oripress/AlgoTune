import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any

class Solver:
    def __init__(self):
        self.directed = False
        self.min_only = False
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the shortest path problem from specified indices using scipy.sparse.csgraph.dijkstra.
        """
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            source_indices = problem["source_indices"]
            if not isinstance(source_indices, list) or not source_indices:
                raise ValueError("source_indices missing or empty")
        except Exception as e:
            return {"distances": []}

        try:
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=source_indices,
                min_only=self.min_only,
            )
        except Exception as e:
            return {"distances": []}

        if dist_matrix.ndim == 1:
            dist_matrix_list = [[(None if np.isinf(d) else d) for d in dist_matrix]]
        else:
            dist_matrix_list = [[(None if np.isinf(d) else d) for d in row] for row in dist_matrix]

        return {"distances": dist_matrix_list}