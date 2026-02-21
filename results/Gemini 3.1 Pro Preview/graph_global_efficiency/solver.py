import numpy as np
import numba as nb
from typing import Any
import itertools
@nb.njit(parallel=True, fastmath=True, boundscheck=False, nogil=True)
def compute_efficiency(n, offsets, edges):
    total_inv_dist = 0.0
    
    for start_node in nb.prange(n):  # pylint: disable=not-an-iterable
        visited = np.zeros(n, dtype=np.uint8)
        queue = np.empty(n, dtype=np.int32)
        
        visited[start_node] = 1
        queue[0] = start_node
        head = 0
        tail = 1
        
        dist = 0
        local_inv_dist = 0.0
        
        while head < tail:
            dist += 1
            inv_dist = 1.0 / dist
            
            old_tail = tail
            for q_idx in range(head, tail):
                u = queue[q_idx]
                
                for i in range(offsets[u], offsets[u+1]):
                    v = edges[i]
                    if not visited[v]:
                        visited[v] = 1
                        queue[tail] = v
                        tail += 1
            
            count = tail - old_tail
            local_inv_dist += count * inv_dist
            head = old_tail
            
        total_inv_dist += local_inv_dist
                        
    return total_inv_dist / (n * (n - 1))

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n <= 1:
            return {"global_efficiency": 0.0}
            
        lengths = np.array([len(neighbors) for neighbors in adj_list], dtype=np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lengths, out=offsets[1:])
        
        num_edges = offsets[-1]
        if num_edges > 0:
            edges = np.fromiter(itertools.chain.from_iterable(adj_list), dtype=np.int32, count=num_edges)
        else:
            edges = np.empty(0, dtype=np.int32)
        
        efficiency = compute_efficiency(n, offsets, edges)
        return {"global_efficiency": float(efficiency)}