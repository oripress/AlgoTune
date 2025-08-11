from __future__ import annotations
import heapq
from typing import Any, List, Dict, Optional

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[Optional[float]]]]:
        """
        Compute shortest‑path distances from each source index using Dijkstra's algorithm.
        The graph is supplied in CSR format (data, indices, indptr, shape).

        Returns:
            {"distances": [[...], [...], ...]}
            Each inner list corresponds to a source node and contains distances to all
            vertices (float) or ``None`` when a vertex is unreachable.
        """
        # ------------------------------------------------------------------
        # Basic validation
        # ------------------------------------------------------------------
        required_keys = {"data", "indices", "indptr", "shape", "source_indices"}
        if not required_keys.issubset(problem):
            return {"distances": []}

        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        shape = problem["shape"]
        source_indices = problem["source_indices"]

        # shape must be square
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            return {"distances": []}
        n = shape[0]
        if shape[1] != n:
            return {"distances": []}

        if not isinstance(source_indices, list) or not source_indices:
            return {"distances": []}

        # ------------------------------------------------------------------
        # Build adjacency list from CSR for fast neighbor iteration
        # ------------------------------------------------------------------
        adjacency: List[List[tuple[int, float]]] = [[] for _ in range(n)]
        # CSR guarantees that len(data) == len(indices) == indptr[-1]
        for u in range(n):
            start = indptr[u]
            end = indptr[u + 1]
            for pos in range(start, end):
                v = indices[pos]
                w = data[pos]
                adjacency[u].append((v, w))

        # ------------------------------------------------------------------
        # Dijkstra implementation for a single source
        # ------------------------------------------------------------------
        def dijkstra(src: int) -> List[Optional[float]]:
            dist = [float('inf')] * n
            dist[src] = 0.0
            heap: List[tuple[float, int]] = [(0.0, src)]
            while heap:
                cur_dist, u = heapq.heappop(heap)
                if cur_dist != dist[u]:
                    continue
                for v, w in adjacency[u]:
                    nd = cur_dist + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            # Convert infinities to None
            return [None if d == float('inf') else d for d in dist]

        # ------------------------------------------------------------------
        # Compute distances for all sources
        # ------------------------------------------------------------------
        result: List[List[Optional[float]]] = []
        for src in source_indices:
            if 0 <= src < n:
                result.append(dijkstra(src))
            else:
                # invalid source index – return a row of Nones
                result.append([None] * n)

        return {"distances": result}