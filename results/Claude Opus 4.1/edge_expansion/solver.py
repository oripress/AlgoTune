class Solver:
    def solve(self, problem, **kwargs):
        """
        Calculates the edge expansion for the given subset S in the graph.
        Edge expansion = |E(S, V-S)| / |S|
        """
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        
        # Handle edge cases
        if n == 0 or not nodes_S_list:
            return {"edge_expansion": 0.0}
            
        # Convert to set for O(1) lookup, filtering out invalid nodes
        nodes_S = set()
        for node in nodes_S_list:
            if 0 <= node < n:
                nodes_S.add(node)
        
        # If S is empty after filtering or contains all nodes
        if not nodes_S or len(nodes_S) == n:
            return {"edge_expansion": 0.0}
        
        # Count edges from S to V-S
        edge_count = 0
        for u in nodes_S:  # Only iterate valid nodes in S
            for v in adj_list[u]:
                if v not in nodes_S:
                    edge_count += 1
        
        # Calculate edge expansion
        expansion = float(edge_count) / float(len(nodes_S))
        return {"edge_expansion": expansion}