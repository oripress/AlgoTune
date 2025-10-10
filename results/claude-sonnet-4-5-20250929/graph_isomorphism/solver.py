class Solver:
    def solve(self, problem):
        """
        Optimized graph isomorphism solver using degree-based heuristics
        and efficient backtracking with forward checking.
        """
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        # Build adjacency sets for O(1) lookup
        adj1 = [set() for _ in range(n)]
        adj2 = [set() for _ in range(n)]
        
        for u, v in edges_g1:
            adj1[u].add(v)
            adj1[v].add(u)
        
        for u, v in edges_g2:
            adj2[u].add(v)
            adj2[v].add(u)
        
        # Compute degrees
        deg1 = [len(adj1[i]) for i in range(n)]
        deg2 = [len(adj2[i]) for i in range(n)]
        
        # Group nodes by degree for quick candidate lookup
        deg_to_nodes2 = {}
        for i in range(n):
            d = deg2[i]
            if d not in deg_to_nodes2:
                deg_to_nodes2[d] = []
            deg_to_nodes2[d].append(i)
        
        # Order nodes by degree (descending) for better pruning
        nodes1_ordered = sorted(range(n), key=lambda x: deg1[x], reverse=True)
        
        mapping = [-1] * n
        reverse_mapping = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True

            u = nodes1_ordered[idx]
            deg_u = deg1[u]

            # Only try nodes with same degree
            if deg_u not in deg_to_nodes2:
                return False

            adj_u = adj1[u]
            
            for v in deg_to_nodes2[deg_u]:
                if reverse_mapping[v] != -1:
                    continue

                # Inline validity check for speed
                adj_v = adj2[v]
                
                # Check all already-mapped neighbors
                valid = True
                for neighbor in adj_u:
                    if mapping[neighbor] != -1:
                        if mapping[neighbor] not in adj_v:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Check reverse: all mapped nodes that are neighbors of v
                # must be mapped from neighbors of u
                for neighbor_v in adj_v:
                    if reverse_mapping[neighbor_v] != -1:
                        if reverse_mapping[neighbor_v] not in adj_u:
                            valid = False
                            break
                
                if not valid:
                    continue

                # Apply mapping
                mapping[u] = v
                reverse_mapping[v] = u

                if backtrack(idx + 1):
                    return True

                # Backtrack
                mapping[u] = -1
                reverse_mapping[v] = -1

            return False
        
        backtrack(0)
        
        return {"mapping": mapping}