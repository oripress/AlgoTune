from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

class Solver:
    def __init__(self, directed: bool = False) -> None:
        """
        Initialize the solver.
        - directed: Whether the graph should be treated as directed. Default False (undirected).
        """
        self.directed = directed

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, List[List[float]]]:
        """
        Compute all-pairs shortest path distances for a graph given in CSR components.

        Input 'problem' keys:
          - data: list[float]
          - indices: list[int]
          - indptr: list[int]
          - shape: [n, n]

        Output:
          - {"distance_matrix": list[list[float|None]]}
            None denotes no path (infinity).
        """
        # Resolve parameters (allow overrides via kwargs)
        directed = kwargs.get("directed", self.directed)

        # Basic validation
        if not all(k in problem for k in ("data", "indices", "indptr", "shape")):
            return {"distance_matrix": []}

        shape = tuple(problem.get("shape", (0, 0)))
        if len(shape) != 2 or shape[0] != shape[1]:
            return {"distance_matrix": []}
        n = int(shape[0])
        if n == 0:
            return {"distance_matrix": []}

        # Fast path: no edges
        try:
            if len(problem["indices"]) == 0 or len(problem["data"]) == 0:
                dist_list = [[0.0 if i == j else float("inf") for j in range(n)] for i in range(n)]
                return {"distance_matrix": dist_list}
        except Exception:
            pass

        # Construct CSR matrix or dense fallback
        try:
            if _HAVE_SCIPY:
                graph_csr = csr_matrix(
                    (
                        np.asarray(problem["data"], dtype=float),
                        np.asarray(problem["indices"], dtype=np.int32),
                        np.asarray(problem["indptr"], dtype=np.int32),
                    ),
                    shape=(n, n),
                )
            else:
                dense = np.full((n, n), np.inf, dtype=float)
                data = problem["data"]
                indices = problem["indices"]
                indptr = problem["indptr"]
                for row in range(n):
                    start, end = indptr[row], indptr[row + 1]
                    for k in range(start, end):
                        col = indices[k]
                        w = float(data[k])
                        if w < dense[row, col]:
                            dense[row, col] = w
                np.fill_diagonal(dense, 0.0)
        except Exception:
            return {"distance_matrix": []}

        # Compute all-pairs shortest paths
        try:
            if _HAVE_SCIPY:
                # Dijkstra is efficient for sparse graphs with non-negative weights
                dist_matrix = csgraph_dijkstra(
                    csgraph=graph_csr, directed=directed, return_predecessors=False
                )
            else:
                dist_matrix = self._all_pairs_dijkstra_dense(dense, directed=directed)
        except Exception:
            return {"distance_matrix": []}

        # Convert to list-of-lists quickly; keep inf values (validator accepts them)
        dist_list: List[List[float]] = dist_matrix.tolist()
        return {"distance_matrix": dist_list}

    @staticmethod
    def _all_pairs_dijkstra_dense(adj: np.ndarray, directed: bool = False) -> np.ndarray:
        """
        Simple Dijkstra for dense adjacency matrix as a fallback when SciPy is unavailable.
        Assumes non-negative weights. Complexity O(n^3) worst-case.
        """
        n = adj.shape[0]
        out = np.empty((n, n), dtype=float)
        for s in range(n):
            dist = np.full(n, np.inf, dtype=float)
            visited = np.zeros(n, dtype=bool)
            dist[s] = 0.0
            for _ in range(n):
                # Find unvisited node with smallest distance
                u = -1
                best = np.inf
                for i in range(n):
                    if not visited[i] and dist[i] < best:
                        best = dist[i]
                        u = i
                if u == -1 or best == np.inf:
                    break
                visited[u] = True
                # Relax edges from u
                du = dist[u]
                row = adj[u]
                for v in range(n):
                    w = row[v]
                    if not np.isfinite(w):
                        continue
                    alt = du + w
                    if alt < dist[v]:
                        dist[v] = alt
            out[s] = dist
        # Ensure zero diagonal
        np.fill_diagonal(out, 0.0)
        return out