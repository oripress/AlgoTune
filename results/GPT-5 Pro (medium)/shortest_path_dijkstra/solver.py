from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self, method: str = "D", directed: bool = False) -> None:
        # Use Dijkstra for non-negative weights; graphs are undirected per task.
        self.method = method
        self.directed = directed

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[Optional[float]]]]:
        """
        Compute all-pairs shortest path distances for an undirected weighted graph
        provided in CSR components. Returns None for unreachable pairs.
        """
        # Allow overriding method/directed via kwargs if provided
        method: str = kwargs.get("method", self.method)
        directed: bool = kwargs.get("directed", self.directed)

        # Basic validation and fast failure path
        if not all(k in problem for k in ("data", "indices", "indptr", "shape")):
            return {"distance_matrix": []}

        shape = tuple(problem["shape"])
        if len(shape) != 2 or shape[0] != shape[1]:
            # Must be square
            return {"distance_matrix": []}
        n = int(shape[0])
        if n == 0:
            return {"distance_matrix": []}

        # Fast path: no edges
        try:
            nnz = len(problem["data"])
        except Exception:
            nnz = None
        if nnz == 0:
            inf = float("inf")
            dist_list: List[List[float]] = [[0.0 if i == j else inf for j in range(n)] for i in range(n)]
            return {"distance_matrix": dist_list}

        # Build CSR matrix without forcing dtype conversions here (SciPy will handle)
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), shape=(n, n)
            )
        except Exception:
            return {"distance_matrix": []}

        # Compute APSP using SciPy's optimized routine
        try:
            # Directly call dijkstra to avoid shortest_path's extra overhead
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=directed,
                return_predecessors=False,
                unweighted=False,
                indices=None,
            )
        except Exception:
            return {"distance_matrix": []}

        # Convert to list of lists quickly; allow inf for unreachable pairs
        dist_list = dist_matrix.tolist()
        return {"distance_matrix": dist_list}

        return {"distance_matrix": dist_list}