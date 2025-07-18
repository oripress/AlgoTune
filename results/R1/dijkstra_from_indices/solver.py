import numpy as np
import heapq
import numba
from numba import njit

@njit
def dijkstra_csr(n, data, indices, indptr, source):
    INF = 1e18
    dist = np.full(n, INF, dtype=np.float64)
    dist[source] = 0.0
    heap = [(0.0, source)]
    visited = np.zeros(n, dtype=numba.boolean)
    
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u] or d != dist[u]:
            continue
        visited[u] = True
        
        start = indptr[u]
        end = indptr[u+1] if u+1 < len(indptr) else len(indices)
        for j in range(start, end):
            v = indices[j]
            if visited[v]:
                continue
            weight = data[j]
            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
                
    return dist

class Solver:
    def __init__(self):
        # Precompile with small graph
        try:
            n = 2
            data_arr = np.array([1.0], dtype=np.float64)
            indices_arr = np.array([1], dtype=np.int32)
            indptr_arr = np.array([0, 1, 1], dtype=np.int32)
            dijkstra_csr(n, data_arr, indices_arr, indptr_arr, 0)
        except:
            pass
            
    def solve(self, problem, **kwargs):
        try:
            n = problem["shape"][0]
            source_indices = problem["source_indices"]
            if not source_indices:
                return {"distances": []}
            
            data_arr = np.array(problem["data"], dtype=np.float64)
            indices_arr = np.array(problem["indices"], dtype=np.int32)
            indptr_arr = np.array(problem["indptr"], dtype=np.int32)
            
            # Run Dijkstra for each source
            distances = []
            for source in source_indices:
                dist_arr = dijkstra_csr(n, data_arr, indices_arr, indptr_arr, source)
                
                # Convert to required format
                row = []
                for d in dist_arr:
                    if d > 1e17:  # INF check
                        row.append(None)
                    else:
                        row.append(float(d))
                distances.append(row)
                
            return {"distances": distances}
        except Exception as e:
            return {"distances": []}