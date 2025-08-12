import numpy as np
from numba import njit
from typing import Any
import concurrent.futures
import heapq

@njit(nogil=True, cache=True)
def dijkstra_linear(data, indices, indptr, n, source):
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    visited = np.zeros(n, dtype=np.bool_)
    
    for _ in range(n):
        u = -1
        min_dist = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
        
        if u == -1:
            break
        
        visited[u] = True
        start = indptr[u]
        end = indptr[u+1]
        
        for i in range(start, end):
            v = indices[i]
            weight = data[i]
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                
    return dist

def dijkstra_heap(data, indices, indptr, n, source):
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    visited = np.zeros(n, dtype=bool)
    heap = [(0.0, source)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        start = indptr[u]
        end = indptr[u+1]
        for i in range(start, end):
            v = indices[i]
            weight = data[i]
            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
                
    return dist

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        n = problem['shape'][0]
        if n == 0:
            return {"distance_matrix": []}
        
        data = np.array(problem['data'], dtype=np.float64)
        indices = np.array(problem['indices'], dtype=np.int32)
        indptr = np.array(problem['indptr'], dtype=np.int32)
        
        # Use heap-based Dijkstra for larger graphs, linear scan for smaller ones
        dijkstra_fn = dijkstra_heap if n > 1000 else dijkstra_linear
        
        # Run in parallel using thread pool
        dist_matrix = np.zeros((n, n))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(n):
                futures.append(executor.submit(dijkstra_fn, data, indices, indptr, n, i))
            
            for i, future in enumerate(futures):
                dist_matrix[i] = future.result()
        
        # Convert to list of lists with None for inf
        dist_matrix = np.where(np.isinf(dist_matrix), None, dist_matrix)
        dist_matrix_list = dist_matrix.tolist()
        
        return {"distance_matrix": dist_matrix_list}