import numba as nb
import numpy as np

@nb.njit
def union_find_count_components(edges, num_nodes):
    """
    JIT-compiled Union-Find with path compression.
    """
    parent = np.arange(num_nodes, dtype=np.int32)
    rank = np.zeros(num_nodes, dtype=np.int32)
    
    # Process all edges
    for i in range(len(edges)):
        u, v = edges[i]
        
        # Find root of u with path compression
        root_u = u
        while parent[root_u] != root_u:
            parent[root_u] = parent[parent[root_u]]  # Path halving
            root_u = parent[root_u]
        
        # Find root of v with path compression
        root_v = v
        while parent[root_v] != root_v:
            parent[root_v] = parent[parent[root_v]]  # Path halving
            root_v = parent[root_v]
        
        # Union by rank
        if root_u != root_v:
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            elif rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_v] = root_u
                rank[root_u] += 1
    
    # Count unique components
    components = 0
    for i in range(num_nodes):
        # Find final root
        root = i
        while parent[root] != root:
            root = parent[root]
        
        if root == i:
            components += 1
    
    return components


class Solver:
    def __init__(self):
        """Pre-compile the Numba function to avoid JIT overhead during solve."""
        # Dummy call to trigger JIT compilation
        dummy_edges = np.array([[0, 1]], dtype=np.int32)
        union_find_count_components(dummy_edges, 2)
    
    def solve(self, problem, **kwargs):
        """
        Count connected components using JIT-compiled Union-Find.
        """
        try:
            num_nodes = problem.get("num_nodes", 0)
            edges = problem["edges"]
            
            # Convert edges to numpy array for Numba - optimize conversion
            if isinstance(edges, np.ndarray):
                edges_array = edges
            else:
                # Fast conversion without copy when possible
                edges_array = np.asarray(edges, dtype=np.int32)
            
            components = union_find_count_components(edges_array, num_nodes)
            
            return {"number_connected_components": int(components)}
            
        except Exception as e:
            return {"number_connected_components": -1}