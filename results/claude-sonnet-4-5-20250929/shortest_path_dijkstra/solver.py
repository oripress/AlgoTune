import numpy as np
import numba
from typing import Any

class Solver:
    def __init__(self):
        # Warmup Numba compilation
        _warmup()

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Fast Dijkstra-based APSP using Numba.
        """
        try:
            n = problem["shape"][0]
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            
            # Compute APSP
            dist_matrix = dijkstra_apsp(data, indices, indptr, n)
            
            # Fast conversion
            mask = np.isinf(dist_matrix)
            dist_matrix_obj = dist_matrix.astype(object)
            dist_matrix_obj[mask] = None
            
            return {"distance_matrix": dist_matrix_obj.tolist()}
        except Exception:
            return {"distance_matrix": []}

def _warmup():
    """Warmup Numba compilation."""
    d = np.array([1.0], dtype=np.float64)
    i = np.array([0], dtype=np.int32)
    p = np.array([0, 1], dtype=np.int32)
    dijkstra_apsp(d, i, p, 1)

@numba.jit(nopython=True, cache=True, fastmath=True)
def dijkstra_apsp(data, indices, indptr, n):
    """All-pairs shortest paths using optimized Dijkstra."""
    result = np.empty((n, n), dtype=np.float64)
    
    for source in range(n):
        dist = np.full(n, np.inf, dtype=np.float64)
        dist[source] = 0.0
        visited = np.zeros(n, dtype=np.bool_)
        
        # Use simple array scan for small graphs, faster than heap overhead
        for _ in range(n):
            # Find minimum unvisited node
            min_dist = np.inf
            u = -1
            for v in range(n):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    u = v
            
            if u == -1:
                break
                
            visited[u] = True
            
            # Relax neighbors
            for idx in range(indptr[u], indptr[u + 1]):
                v = indices[idx]
                if not visited[v]:
                    new_dist = dist[u] + data[idx]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
        
        result[source] = dist
    
    return result