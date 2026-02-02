from typing import Any
import numpy as np
from numba import njit

@njit(cache=True)
def union_find_count_flat(n, edges_flat, m):
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    num_components = n
    
    for i in range(m):
        u = edges_flat[2*i]
        v = edges_flat[2*i + 1]
        
        # Find root of u with path halving
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        root_u = u
        
        # Find root of v with path halving
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        root_v = v
        
        # Union by rank
        if root_u != root_v:
            num_components -= 1
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            elif rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    return num_components

class Solver:
    def __init__(self):
        # Pre-allocate a buffer for edges
        self._edges_buffer = np.empty(100000, dtype=np.int32)
        # Warm up the JIT
        dummy = np.array([0, 1], dtype=np.int32)
        union_find_count_flat(2, dummy, 1)
    
    def solve(self, problem, **kwargs) -> Any:
        n = problem.get("num_nodes", 0)
        edges = problem["edges"]
        
        if n == 0:
            return {"number_connected_components": 0}
        
        m = len(edges)
        if m == 0:
            return {"number_connected_components": n}
        
        # Use buffer if possible, otherwise allocate
        needed = 2 * m
        if needed <= len(self._edges_buffer):
            edges_flat = self._edges_buffer[:needed]
        else:
            edges_flat = np.empty(needed, dtype=np.int32)
        
        # Fast conversion
        idx = 0
        for u, v in edges:
            edges_flat[idx] = u
            edges_flat[idx + 1] = v
            idx += 2
        
        num_components = union_find_count_flat(n, edges_flat, m)
        
        return {"number_connected_components": num_components}