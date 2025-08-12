import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List, Union

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[Union[float, None]]]]:
        """
        Solves the all-pairs shortest path problem using scipy.sparse.csgraph.shortest_path.
        
        :param problem: A dictionary representing the graph in CSR components.
        :return: A dictionary with key "distance_matrix":
                 "distance_matrix": The matrix of shortest path distances (list of lists).
                                     None indicates no path.
        """
        try:
            # Create CSR matrix from the problem data
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), 
                shape=problem["shape"],
                dtype=np.float64
            )
        except Exception:
            return {"distance_matrix": []}  # Indicate failure

        try:
            # Compute all-pairs shortest paths using scipy's optimized implementation
            # For sparse graphs, 'FW' (Floyd-Warshall) might be more efficient than 'auto'
            dist_matrix = scipy.sparse.csgraph.shortest_path(
                csgraph=graph_csr, 
                method='FW',  # Use Floyd-Warshall algorithm
                directed=False,  # The graph is undirected
                return_predecessors=False,  # We only need distances
                unweighted=False  # Use actual edge weights
            )
        except Exception:
            return {"distance_matrix": []}  # Indicate failure
        # Convert to the required output format more efficiently
        # Create a mask for infinite values
        mask = np.isinf(dist_matrix)
        # Use the mask to replace inf with None
        dist_matrix_list = dist_matrix.astype(object)
        dist_matrix_list[mask] = None
        dist_matrix_list = dist_matrix_list.tolist()
        
        return {"distance_matrix": dist_matrix_list}