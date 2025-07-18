from typing import Any
import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def find(parent, x):
    """Find with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression
    while x != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root

@nb.njit(cache=True, fastmath=True)
def count_components_numba(edges, num_nodes):
    """Count connected components using optimized Union-Find."""
    parent = np.arange(num_nodes, dtype=np.int32)
    
    # Process edges
    for i in range(edges.shape[0]):
        u, v = edges[i, 0], edges[i, 1]
        root_u = find(parent, u)
        root_v = find(parent, v)
        if root_u != root_v:
            if root_u < root_v:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
    
    # Count components
    components = 0
    for i in range(num_nodes):
        if find(parent, i) == i:
            components += 1
    
    return components

class Solver:
    def __init__(self):
        # Pre-compile numba functions
        dummy = np.array([[0, 1]], dtype=np.int32)
        count_components_numba(dummy, 2)
    
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Count connected components using Numba-optimized Union-Find."""
        num_nodes = problem.get("num_nodes", 0)
        edges = problem.get("edges", [])
        
        if num_nodes == 0:
            return {"number_connected_components": 0}
        
        # Convert edges to numpy array
        if edges:
            edges_array = np.array(edges, dtype=np.int32)
        else:
            edges_array = np.empty((0, 2), dtype=np.int32)
        
        # Use Numba implementation
        num_components = count_components_numba(edges_array, num_nodes)
        
        return {"number_connected_components": num_components}