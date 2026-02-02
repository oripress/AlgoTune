import numpy as np
from numba import njit
from typing import Any

@njit(cache=True)
def _compute_efficiency_numba(neighbors, indptr, n):
    """Compute global efficiency using BFS from each node."""
    total_inv = 0.0
    
    for source in range(n):
        # BFS from source
        dist = np.full(n, -1, dtype=np.int32)
        dist[source] = 0
        queue = np.empty(n, dtype=np.int32)
        head = 0
        tail = 1
        queue[0] = source
        
        while head < tail:
            u = queue[head]
            head += 1
            for i in range(indptr[u], indptr[u + 1]):
                v = neighbors[i]
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    total_inv += 1.0 / dist[v]
                    queue[tail] = v
                    tail += 1
    
    return total_inv / (n * (n - 1))

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Convert to CSR-like format for numba
        total_edges = sum(len(nbrs) for nbrs in adj_list)
        if total_edges == 0:
            return {"global_efficiency": 0.0}
        
        neighbors = np.empty(total_edges, dtype=np.int32)
        indptr = np.empty(n + 1, dtype=np.int32)
        indptr[0] = 0
        idx = 0
        for i, nbrs in enumerate(adj_list):
            for v in nbrs:
                neighbors[idx] = v
                idx += 1
            indptr[i + 1] = idx
        
        efficiency = _compute_efficiency_numba(neighbors, indptr, n)
        return {"global_efficiency": float(efficiency)}