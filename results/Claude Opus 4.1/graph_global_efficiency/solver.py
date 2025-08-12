import numpy as np
from collections import deque
from numba import njit

class Solver:
    def solve(self, problem, **kwargs):
        """Calculate global efficiency of a graph using optimized BFS."""
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        # Handle edge cases
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Convert adjacency list to CSR-like format for efficient access
        # First, count total edges
        total_edges = sum(len(neighbors) for neighbors in adj_list)
        
        # Create arrays for CSR representation
        edges = np.zeros(total_edges, dtype=np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        
        idx = 0
        for i, neighbors in enumerate(adj_list):
            offsets[i] = idx
            for neighbor in neighbors:
                edges[idx] = neighbor
                idx += 1
        offsets[n] = idx
        
        # Use JIT-compiled function for efficiency calculation
        efficiency = compute_global_efficiency_jit(edges, offsets, n)
        
        return {"global_efficiency": float(efficiency)}

@njit
def compute_global_efficiency_jit(edges, offsets, n):
    """JIT-compiled function to compute global efficiency using BFS."""
    if n <= 1:
        return 0.0
    
    total_inverse_distance = 0.0
    
    # Pre-allocate arrays for BFS
    distances = np.empty(n, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    
    # Compute shortest paths from each node using BFS
    for source in range(n):
        # Initialize distances
        distances[:] = -1
        distances[source] = 0
        
        # Initialize queue
        queue[0] = source
        front = 0
        rear = 1
        
        # BFS
        while front < rear:
            u = queue[front]
            front += 1
            
            # Iterate through neighbors of u
            for edge_idx in range(offsets[u], offsets[u + 1]):
                v = edges[edge_idx]
                if distances[v] == -1:
                    distances[v] = distances[u] + 1
                    queue[rear] = v
                    rear += 1
        
        # Add inverse distances
        for target in range(n):
            if target != source and distances[target] > 0:
                total_inverse_distance += 1.0 / distances[target]
    
    # Calculate global efficiency
    efficiency = total_inverse_distance / (n * (n - 1))
    return efficiency