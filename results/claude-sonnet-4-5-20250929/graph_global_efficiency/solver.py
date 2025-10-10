from collections import deque
import numba
import numpy as np

@numba.njit(fastmath=True, cache=True)
def compute_global_efficiency(adj_list_array, adj_offsets, n):
    """
    Compute global efficiency using Numba-optimized BFS with early termination.
    """
    total_inv_distance = 0.0
    
    for start in range(n):
        # BFS with early termination
        visited = np.zeros(n, dtype=np.bool_)
        distance = np.zeros(n, dtype=np.int32)
        queue = np.empty(n, dtype=np.int32)
        queue_start = 0
        queue_end = 0
        
        queue[queue_end] = start
        queue_end += 1
        visited[start] = True
        visited_count = 1
        
        while queue_start < queue_end:
            u = queue[queue_start]
            queue_start += 1
            
            # Get neighbors from flattened array
            start_idx = adj_offsets[u]
            end_idx = adj_offsets[u + 1]
            
            for idx in range(start_idx, end_idx):
                v = adj_list_array[idx]
                if not visited[v]:
                    visited[v] = True
                    distance[v] = distance[u] + 1
                    queue[queue_end] = v
                    queue_end += 1
                    visited_count += 1
                    total_inv_distance += 1.0 / distance[v]
                    
                    # Early termination if all nodes visited
                    if visited_count == n:
                        break
            
            if visited_count == n:
                break
    
    return total_inv_distance / (n * (n - 1))

class Solver:
    def solve(self, problem, **kwargs):
        """
        Calculate global efficiency using Numba-optimized BFS.
        
        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}
        
        Returns:
            A dictionary containing the global efficiency.
            {"global_efficiency": efficiency_value}
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        # Handle edge cases
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Flatten adjacency list for Numba
        adj_list_flat = []
        adj_offsets = [0]
        for neighbors in adj_list:
            adj_list_flat.extend(neighbors)
            adj_offsets.append(len(adj_list_flat))
        
        adj_list_array = np.array(adj_list_flat, dtype=np.int32)
        adj_offsets_array = np.array(adj_offsets, dtype=np.int32)
        
        efficiency = compute_global_efficiency(adj_list_array, adj_offsets_array, n)
        
        return {"global_efficiency": float(efficiency)}