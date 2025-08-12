from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Ultra-optimized backtracking with tuple unpacking and minimal overhead.
        """
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        # Build adjacency matrices for O(1) lookup
        adj1 = [[False] * n for _ in range(n)]
        adj2 = [[False] * n for _ in range(n)]
        
        for u, v in edges_g1:
            adj1[u][v] = True
            adj1[v][u] = True
            
        for u, v in edges_g2:
            adj2[u][v] = True
            adj2[v][u] = True
        
        # Calculate degrees
        degrees1 = [0] * n
        degrees2 = [0] * n
        for i in range(n):
            degrees1[i] = sum(adj1[i])
            degrees2[i] = sum(adj2[i])
        
        # Check degree sequence compatibility
        if sorted(degrees1) != sorted(degrees2):
            return {"mapping": [-1] * n}
        
        # Pre-group nodes by degree for faster candidate selection
        degree_candidates = {}
        for i in range(n):
            d = degrees2[i]
            if d not in degree_candidates:
                degree_candidates[d] = []
            degree_candidates[d].append(i)
        
        # Order nodes by degree (highest degree first for better pruning)
        node_order = sorted(range(n), key=lambda i: -degrees1[i])
        
        # Initialize mapping
        mapping = [-1] * n
        reverse_mapping = [-1] * n
        
        # Use local variables for faster access
        _adj1 = adj1
        _adj2 = adj2
        _mapping = mapping
        _reverse_mapping = reverse_mapping
        _node_order = node_order
        
        def backtrack(pos):
            if pos == n:
                return True
                
            u = _node_order[pos]
            d = degrees1[u]
            
            # Try only candidates with the right degree
            candidates = degree_candidates[d]
            for v in candidates:
                if _reverse_mapping[v] == -1:
                    # Check if mapping u to v preserves adjacency
                    # Pre-fetch adjacency rows for faster access
                    u_row = _adj1[u]
                    v_row = _adj2[v]
                    valid = True
                    for i in range(pos):
                        prev_u = _node_order[i]
                        prev_v = _mapping[prev_u]
                        if u_row[prev_u] != v_row[prev_v]:
                            valid = False
                            break
                    
                    if valid:
                        _mapping[u] = v
                        _reverse_mapping[v] = u
                        
                        if backtrack(pos + 1):
                            return True
                            
                        _mapping[u] = -1
                        _reverse_mapping[v] = -1
            
            return False
        
        if backtrack(0):
            return {"mapping": mapping}
        else:
            return {"mapping": [-1] * n}