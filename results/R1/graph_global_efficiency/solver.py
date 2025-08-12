import numpy as np
from numba import jit

@jit(nopython=True)
def compute_total(n, indptr, indices):
    total = 0.0
    # For each node as source
    for u in range(n):
        # BFS setup
        dist = np.full(n, -1, dtype=np.int32)
        queue = np.empty(n, dtype=np.int32)
        q_start = 0
        q_end = 0
        
        # Initialize BFS from u
        queue[q_end] = u
        q_end += 1
        dist[u] = 0
        
        # Process queue
        while q_start < q_end:
            current = queue[q_start]
            q_start += 1
            start_idx = indptr[current]
            end_idx = indptr[current+1]
            
            # Visit neighbors
            for idx in range(start_idx, end_idx):
                neighbor = indices[idx]
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[current] + 1
                    queue[q_end] = neighbor
                    q_end += 1
        
        # Sum reciprocal distances
        for v in range(n):
            if v != u and dist[v] != -1:
                total += 1.0 / dist[v]
                
    return total

class Solver:
    def __init__(self):
        # Trigger JIT compilation with small graph
        adj_list = [[1], [0]]
        n = 2
        indptr = [0, 1, 2]
        indices = [1, 0]
        indptr_np = np.array(indptr, dtype=np.int32)
        indices_np = np.array(indices, dtype=np.int32)
        compute_total(n, indptr_np, indices_np)
    
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Build CSR representation
        indptr = [0]
        indices = []
        for neighbors in adj_list:
            indices.extend(neighbors)
            indptr.append(len(indices))
        
        # Convert to numpy arrays
        indptr_np = np.array(indptr, dtype=np.int32)
        indices_np = np.array(indices, dtype=np.int32)
        
        # Compute total reciprocal distance
        total = compute_total(n, indptr_np, indices_np)
        global_eff = total / (n * (n - 1))
        
        return {"global_efficiency": global_eff}