class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        
        # Handle edge cases
        if n == 0 or not nodes_S_list:
            return {"edge_expansion": 0.0}
        
        size_S = len(nodes_S_list)
        if size_S == n:
            return {"edge_expansion": 0.0}
        
        # Convert to set for O(1) lookup
        nodes_S = set(nodes_S_list)
        
        # Count edges crossing S boundary (both directions)
        # Edges from S to V-S and from V-S to S
        edge_count = 0
        
        # Count edges from S to V-S
        for u in nodes_S:
            for v in adj_list[u]:
                if v not in nodes_S:
                    edge_count += 1
        
        # Count edges from V-S to S
        for u in range(n):
            if u not in nodes_S:
                for v in adj_list[u]:
                    if v in nodes_S:
                        edge_count += 1
        
        size_VS = n - size_S
        expansion = edge_count / min(size_S, size_VS)
        return {"edge_expansion": expansion}