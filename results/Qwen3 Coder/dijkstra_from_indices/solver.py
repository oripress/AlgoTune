import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List, Union
from numpy import isinf

class Solver:
    def __init__(self):
        self.directed = False
        self.min_only = False

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[Union[float, None]]]]:
        """
        Solves the shortest path problem from specified indices using scipy.sparse.csgraph.dijkstra.

        :param problem: A dictionary representing the graph (CSR) and source indices.
        :return: A dictionary with key "distances": shortest path distances from the source nodes.
        """
        try:
            # Create CSR matrix from the provided data
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )

            source_indices = problem["source_indices"]

            # Check if source_indices is valid
            if not isinstance(source_indices, list) or not source_indices:
                return {"distances": []}

            # Use scipy's dijkstra implementation
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=source_indices,
                min_only=self.min_only,
            )

            # Convert to the required format
            if dist_matrix.ndim == 1:
                # Single source case
                dist_matrix_list = [[(None if isinf(d) else float(d)) for d in dist_matrix]]
            else:
                # Multiple sources case
                dist_matrix_list = [[(None if isinf(d) else float(d)) for d in row] for row in dist_matrix]

            return {"distances": dist_matrix_list}

        except Exception:
            return {"distances": []}