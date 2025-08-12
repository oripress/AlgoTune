import numpy as np
from typing import Any, Dict, List
import networkx as nx

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Optimized graph isomorphism solver using adjacency matrix representation
        and backtracking with pruning.
        """
        n = problem["num_nodes"]
        
        # Build adjacency matrices
        adj1 = np.zeros((n, n), dtype=bool)
        adj2 = np.zeros((n, n), dtype=bool)
        
        for u, v in problem["edges_g1"]:
            adj1[u, v] = True
            adj1[v, u] = True
            
        for x, y in problem["edges_g2"]:
            adj2[x, y] = True
            adj2[y, x] = True
        
        # Degree-based node ordering for better pruning
        degrees1 = np.sum(adj1, axis=1)
        degrees2 = np.sum(adj2, axis=1)
        
        # Sort nodes by degree (for both graphs) - use reverse order for G1 to process high-degree nodes first
        nodes1_sorted = np.argsort(degrees1)[::-1]
        nodes2_sorted = np.argsort(degrees2)
        
        # Create degree groups for more efficient matching
        degree_groups1 = {}
        degree_groups2 = {}
        
        for i in range(n):
            deg1 = degrees1[i]
            deg2 = degrees2[i]
            if deg1 not in degree_groups1:
                degree_groups1[deg1] = []
            if deg2 not in degree_groups2:
                degree_groups2[deg2] = []
            degree_groups1[deg1].append(i)
            degree_groups2[deg2].append(i)
        
        # Mapping array: mapping[i] = j means node i in G1 maps to node j in G2
        # Mapping array: mapping[i] = j means node i in G1 maps to node j in G2
        mapping = [-1] * n
        used = np.zeros(n, dtype=bool)
        def is_valid_mapping(u, v):
            """Check if mapping node u to node v is valid so far"""
            # Check if v is already used
            if used[v]:
                return False

            # Check adjacency consistency with already mapped nodes
            # Precompute neighbors of u in G1 for faster access
            u_neighbors = adj1[u]
            for i in range(n):
                if mapping[i] != -1:  # Already mapped node
                    # If i and u are connected in G1, then mapping[i] and v should be connected in G2
                    if u_neighbors[i] != adj2[v, mapping[i]]:
                        return False
            return True
        
        def backtrack(index):
            """Backtracking search for isomorphism"""
            if index == n:
                return True
                
            # Get current node from G1 (using degree-based ordering)
            u = nodes1_sorted[index]
            
            # Get possible mappings based on degree
            u_degree = degrees1[u]
            if u_degree in degree_groups2:
                candidates = degree_groups2[u_degree]
            else:
                candidates = range(n)
            
            for v in candidates:
                if is_valid_mapping(u, v):
                    mapping[u] = v
                    used[v] = True
                    
                    if backtrack(index + 1):
                        return True
                        
                    mapping[u] = -1
                    used[v] = False
            
            return False
        
        # Start the search
        if backtrack(0):
            return {"mapping": mapping}
        else:
            # Fallback to NetworkX if our method fails (shouldn't happen for valid inputs)
            G1 = nx.Graph()
            G2 = nx.Graph()
            G1.add_nodes_from(range(n))
            G2.add_nodes_from(range(n))
            
            for u, v in problem["edges_g1"]:
                G1.add_edge(u, v)
            for x, y in problem["edges_g2"]:
                G2.add_edge(x, y)
            
            gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
            iso_map = next(gm.isomorphisms_iter())
            mapping = [iso_map[u] for u in range(n)]
            return {"mapping": mapping}