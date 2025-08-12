from typing import Any, Dict, List
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange

@njit
def _bfs_total_inv(num_nodes: int, indptr: np.ndarray, indices: np.ndarray) -> float:
    """
    Perform BFS from every source node and accumulate the sum of
    1 / shortest_path_length for all reachable pairs (u, v), u != v.
    """
    total = 0.0
    # Pre‑allocate arrays reused for each source
    dist = np.empty(num_nodes, dtype=np.int32)
    visited = np.empty(num_nodes, dtype=np.int32)
    queue = np.empty(num_nodes, dtype=np.int32)

    token = 1  # visitation token, incremented each BFS
    for src in range(num_nodes):
        token += 1
        head = 0
        tail = 0
        queue[tail] = src
        tail += 1
        dist[src] = 0
        visited[src] = token

        while head < tail:
            u = queue[head]
            head += 1
            du = dist[u] + 1
            for idx in range(indptr[u], indptr[u + 1]):
                v = indices[idx]
                if visited[v] != token:
                    visited[v] = token
                    dist[v] = du
                    queue[tail] = v
                    tail += 1
                    total += 1.0 / du
    return total

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> Dict[str, float]:
        """
        Compute the global efficiency of an undirected, unweighted graph.

        Global efficiency is the average of the inverse shortest path lengths
        over all pairs of distinct nodes. For disconnected pairs the contribution
        is zero (treated as infinite distance).
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Edge cases: empty graph or single node have efficiency 0.
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Build CSR adjacency matrix (undirected, without duplicate edges)
        rows = []
        cols = []
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u <= v:          # add each undirected edge once
                    rows.append(u)
                    cols.append(v)
                    rows.append(v)
                    cols.append(u)

        data = np.ones(len(rows), dtype=np.int8)
        csr = csr_matrix((data, (rows, cols)), shape=(n, n))

        # Use the compiled Numba BFS to compute the sum of inverse distances
        total_inv = _bfs_total_inv(n, csr.indptr, csr.indices)

        efficiency = total_inv / (n * (n - 1))
        return {"global_efficiency": float(efficiency)}