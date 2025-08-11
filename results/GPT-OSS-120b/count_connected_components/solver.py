from typing import Any, Dict, List, Tuple

import numpy as np
import numba

@numba.njit(cache=True)
def _count_components(n: int, edges: np.ndarray) -> int:
    """
    Numba‑accelerated Union‑Find that returns the number of connected components.
    """
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    m = edges.shape[0]
    for i in range(m):
        u = edges[i, 0]
        v = edges[i, 1]

        # find root of u
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]

        # find root of v
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]

        if u == v:
            continue

        # union by size
        if size[u] < size[v]:
            u, v = v, u
        parent[v] = u
        size[u] += size[v]

    # final count of roots (each component has exactly one root where parent[i] == i)
    cnt = int((parent == np.arange(n, dtype=parent.dtype)).sum())
    return cnt
class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, int]:
        """
        Count the number of connected components in an undirected graph.
        """
        try:
            n = int(problem.get("num_nodes", 0))
            edges = problem.get("edges", [])
            # Convert edge list to a NumPy int32 array of shape (m, 2)
            if not edges:
                # No edges: each node is its own component
                return {"number_connected_components": n}
            edges_arr = np.array(edges, dtype=np.int32)
            # Ensure correct shape
            if edges_arr.ndim == 1:
                edges_arr = edges_arr.reshape(-1, 2)

            cc = _count_components(n, edges_arr)
            return {"number_connected_components": int(cc)}
        except Exception:
            return {"number_connected_components": -1}