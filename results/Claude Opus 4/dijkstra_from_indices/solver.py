import numpy as np
import numba
from numba import njit
from typing import Any

@njit
def dijkstra_from_csr(n, data, indices, indptr, source):
    """Direct Dijkstra on CSR format using numba JIT compilation."""
    INF = 1e20
    dist = np.full(n, INF, dtype=np.float64)
    dist[source] = 0.0
    visited = np.zeros(n, dtype=np.bool_)
    
    # Simple priority queue using arrays
    heap_dist = np.full(n, INF, dtype=np.float64)
    heap_node = np.arange(n, dtype=np.int32)
    heap_size = 0
    
    # Add source to heap
    heap_dist[0] = 0.0
    heap_node[0] = source
    heap_size = 1
    
    while heap_size > 0:
        # Find minimum in heap (linear search for simplicity)
        min_idx = 0
        min_dist = heap_dist[0]
        for i in range(1, heap_size):
            if heap_dist[i] < min_dist:
                min_dist = heap_dist[i]
                min_idx = i
        
        # Extract minimum
        u = heap_node[min_idx]
        d = heap_dist[min_idx]
        
        # Remove from heap by swapping with last element
        heap_size -= 1
        if min_idx < heap_size:
            heap_dist[min_idx] = heap_dist[heap_size]
            heap_node[min_idx] = heap_node[heap_size]
        
        if visited[u]:
            continue
            
        visited[u] = True
        
        # Process neighbors from CSR
        start = indptr[u]
        end = indptr[u + 1]
        
        for idx in range(start, end):
            v = indices[idx]
            w = data[idx]
            
            if not visited[v]:
                alt = d + w
                if alt < dist[v]:
                    dist[v] = alt
                    
                    # Add to heap
                    heap_dist[heap_size] = alt
                    heap_node[heap_size] = v
                    heap_size += 1
    
    return dist

class Solver:
    def __init__(self):
        # Warm up numba JIT compilation
        try:
            dummy_data = np.array([1.0, 1.0], dtype=np.float64)
            dummy_indices = np.array([1, 0], dtype=np.int32)
            dummy_indptr = np.array([0, 1, 2], dtype=np.int32)
            dijkstra_from_csr(2, dummy_data, dummy_indices, dummy_indptr, 0)
        except:
            pass
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the shortest path problem using optimized Dijkstra implementation.
        """
        try:
            # Convert to numpy arrays with appropriate dtypes
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            n = problem["shape"][0]
            source_indices = problem["source_indices"]
            
            if not isinstance(source_indices, list) or not source_indices:
                return {"distances": []}
        except Exception:
            return {"distances": []}
        
        try:
            distances = []
            for source in source_indices:
                dist = dijkstra_from_csr(n, data, indices, indptr, source)
                # Convert to list with None for infinity
                dist_list = []
                for d in dist:
                    if d >= 1e20:
                        dist_list.append(None)
                    else:
                        dist_list.append(float(d))
                distances.append(dist_list)
            
            return {"distances": distances}
        except Exception:
            return {"distances": []}