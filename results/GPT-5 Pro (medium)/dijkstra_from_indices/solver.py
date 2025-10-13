from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback if SciPy not available
    _SCIPY_AVAILABLE = False

class Solver:
    def __init__(self) -> None:
        # Defaults align with the task description (undirected graph, full distances)
        self.default_directed = False

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, List[List[Optional[float]]]]:
        """
        Compute shortest path distances from specified source nodes in a weighted, sparse graph.

        Inputs:
          - problem: dict with keys "data", "indices", "indptr", "shape", "source_indices"
          - kwargs: optional overrides:
              directed: bool (default False)
              min_only: bool (default False) -> if True, returns one row (min across sources)

        Output:
          - {"distances": List[List[float|None]]}
            Distances with None representing infinity.
        """
        # Validate and extract inputs
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            source_indices = problem["source_indices"]
        except Exception:
            return {"distances": []}

        if not isinstance(source_indices, list) or len(source_indices) == 0:
            return {"distances": []}

        # Options
        directed = bool(kwargs.get("directed", problem.get("directed", self.default_directed)))
        min_only = bool(kwargs.get("min_only", problem.get("min_only", False)))

        # Construct CSR graph
        try:
            # Ensure proper dtypes to minimize overhead
            data_arr = np.asarray(data, dtype=float)
            indices_arr = np.asarray(indices, dtype=np.int64)
            indptr_arr = np.asarray(indptr, dtype=np.int64)
            n_rows, n_cols = int(shape[0]), int(shape[1])
            graph_csr = csr_matrix((data_arr, indices_arr, indptr_arr), shape=(n_rows, n_cols)) if _SCIPY_AVAILABLE else None
        except Exception:
            # Bad graph definition
            return {"distances": []}

        # Solve using SciPy if available
        if _SCIPY_AVAILABLE:
            try:
                dist = _sp_dijkstra(
                    csgraph=graph_csr,
                    directed=directed,
                    indices=source_indices,
                    min_only=min_only,
                    return_predecessors=False,
                )
            except Exception:
                return {"distances": []}

            # Normalize to 2D array of shape (num_rows, n)
            if np.ndim(dist) == 1:
                dist = dist[np.newaxis, :]

            # Convert to required list format with None for infinity
            out: List[List[Optional[float]]] = []
            isfinite = np.isfinite
            for i in range(dist.shape[0]):
                row = dist[i]
                out.append([None if not isfinite(x) else float(x) for x in row])
            return {"distances": out}

        # Fallback: pure Python Dijkstra using heapq on CSR graph
        # Note: Assumes non-negative weights. For directed=False, expects CSR to already contain
        # both directions (typical for undirected CSR adjacency).
        import heapq

        n = int(shape[0])
        indptr_arr = np.asarray(indptr, dtype=np.int64)
        indices_arr = np.asarray(indices, dtype=np.int64)
        data_arr = np.asarray(data, dtype=float)

        def dijkstra_single(src: int) -> np.ndarray:
            dist = np.full(n, np.inf, dtype=float)
            dist[src] = 0.0
            visited = np.zeros(n, dtype=np.uint8)
            pq: List[tuple[float, int]] = [(0.0, src)]
            heappop = heapq.heappop
            heappush = heapq.heappush

            while pq:
                d, u = heappop(pq)
                if visited[u]:
                    continue
                visited[u] = 1
                # If this popped distance is larger than the current best, skip
                if d > dist[u]:
                    continue
                start = indptr_arr[u]
                end = indptr_arr[u + 1]
                for k in range(start, end):
                    v = indices_arr[k]
                    nd = d + data_arr[k]
                    if nd < dist[v]:
                        dist[v] = nd
                        heappush(pq, (nd, v))
            return dist

        try:
            if min_only and len(source_indices) > 1:
                best = np.full(n, np.inf, dtype=float)
                for s in source_indices:
                    d = dijkstra_single(int(s))
                    # elementwise min
                    np.minimum(best, d, out=best)
                rows = [best]
            else:
                rows = [dijkstra_single(int(s)) for s in source_indices]
        except Exception:
            return {"distances": []}

        # Convert to required list-of-lists with None for infinity
        out: List[List[Optional[float]]] = []
        isfinite = np.isfinite
        for row in rows:
            out.append([None if not isfinite(x) else float(x) for x in row])

        return {"distances": out}