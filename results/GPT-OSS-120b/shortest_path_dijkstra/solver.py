import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[Any]]]:
        """
        Compute all‑pairs shortest‑path distances for an undirected weighted graph
        given in CSR format.

        Parameters
        ----------
        problem : dict
            Must contain the CSR components:
            - "data":   list of edge weights
            - "indices": list of column indices for each weight
            - "indptr": list of index pointers into data/indices
            - "shape":  [n, n] where n is the number of nodes

        Returns
        -------
        dict
            {"distance_matrix": [[...], ...]}
            Entries are floats; unreachable pairs are represented by ``None``.
        """
        # Validate required keys
        for key in ("data", "indices", "indptr", "shape"):
            if key not in problem:
                raise ValueError(f"Missing key '{key}' in problem definition")

        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        shape = problem["shape"]
        n = shape[0]

        # Convert to NumPy arrays for efficient CSR construction
        data_arr = np.asarray(data, dtype=float)
        indices_arr = np.asarray(indices, dtype=int)
        indptr_arr = np.asarray(indptr, dtype=int)

        # Build CSR matrix
        csr = sp.csr_matrix((data_arr, indices_arr, indptr_arr), shape=shape)

        # Use SciPy's auto‑selected algorithm for best performance on any graph size.
        # It chooses the most efficient method (Floyd‑Warshall, Dijkstra, etc.) internally.
        dist = csgraph.shortest_path(csgraph=csr, directed=False, method='auto')
        distance_matrix = np.where(np.isinf(dist), None, dist).tolist()
        return {"distance_matrix": distance_matrix}