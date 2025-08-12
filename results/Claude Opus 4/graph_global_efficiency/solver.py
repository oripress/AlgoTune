from typing import Any
import numba
import numpy as np

@numba.njit(cache=True)
def bfs_all_pairs_optimized(adj_array, adj_lengths, n):
    """Compute sum of inverse distances using optimized BFS from all nodes."""
    total_inverse_distance = 0.0
    
    # Pre-allocate arrays that will be reused
    distances = np.empty(n, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    
    for start in range(n):
        # Reset distances array
        distances.fill(-1)
        distances[start] = 0
        
        # BFS using pre-allocated queue
        queue[0] = start
        queue_start = 0
        queue_end = 1
        
        while queue_start < queue_end:
            u = queue[queue_start]
            queue_start += 1
            current_dist = distances[u]
            next_dist = current_dist + 1
            
            # Iterate through neighbors using actual length
            neighbor_count = adj_lengths[u]
            for i in range(neighbor_count):
                v = adj_array[u, i]
                if distances[v] == -1:
                    distances[v] = next_dist
                    queue[queue_end] = v
                    queue_end += 1
        
        # Add inverse distances to total
        for end in range(n):
            if end != start:
                dist = distances[end]
                if dist > 0:
                    total_inverse_distance += 1.0 / dist
    
    return total_inverse_distance

class Solver:
    def __init__(self):
        # Pre-compile the function with a dummy call
        dummy_adj = np.zeros((2, 1), dtype=np.int32)
        dummy_lengths = np.array([1, 1], dtype=np.int32)
        _ = bfs_all_pairs_optimized(dummy_adj, dummy_lengths, 2)
    
    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> Any:
        """Calculate global efficiency of an undirected graph."""
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        # Handle edge cases
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Convert adjacency list to numpy arrays
        adj_lengths = np.array([len(neighbors) for neighbors in adj_list], dtype=np.int32)
        max_degree = np.max(adj_lengths) if n > 0 else 0
        
        if max_degree == 0:
            return {"global_efficiency": 0.0}
        
        adj_array = np.zeros((n, max_degree), dtype=np.int32)
        for i, neighbors in enumerate(adj_list):
            if neighbors:
                adj_array[i, :len(neighbors)] = neighbors
        
        # Calculate sum of inverse distances
        total_inverse_distance = bfs_all_pairs_optimized(adj_array, adj_lengths, n)
        
        # Calculate global efficiency
        global_efficiency = total_inverse_distance / (n * (n - 1))
        
        return {"global_efficiency": global_efficiency}