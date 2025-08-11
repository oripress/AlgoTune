import numba
import numpy as np

class Solver:
    def __init__(self):
        # Pre-compile the numba function
        self._solve_jit = self._create_jit_solver()
    
    def _create_jit_solver(self):
        @numba.njit(cache=True, fastmath=True, parallel=False)
        def count_components(num_nodes, edges_flat, num_edges):
            if num_nodes == 0:
                return 0
            
            parent = np.arange(num_nodes, dtype=np.int32)
            
            # Start with num_nodes components
            components = num_nodes
            
            for i in range(num_edges):
                u = edges_flat[i * 2]
                v = edges_flat[i * 2 + 1]
                
                # Find root of u with path halving
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                
                # Find root of v with path halving
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                
                # Union
                if u != v:
                    parent[v] = u
                    components -= 1
            
            return components
        
        return count_components
    
    def solve(self, problem, **kwargs):
        """Count connected components using Numba-optimized Union-Find."""
        num_nodes = problem.get("num_nodes", 0)
        edges = problem.get("edges", [])
        
        if num_nodes == 0:
            return {"number_connected_components": 0}
        
        # Flatten edges for better memory access pattern
        if edges:
            edges_flat = np.array(edges, dtype=np.int32).flatten()
            num_edges = len(edges)
        else:
            edges_flat = np.empty(0, dtype=np.int32)
            num_edges = 0
        
        components = self._solve_jit(num_nodes, edges_flat, num_edges)
        
        return {"number_connected_components": int(components)}