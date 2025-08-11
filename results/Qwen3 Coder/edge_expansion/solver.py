class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Calculates the edge expansion for the given subset S in the graph.
        
        Args:
            problem: A dictionary containing "adjacency_list" and "nodes_S".
            
        Returns:
            A dictionary containing the edge expansion value.
        """
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        nodes_S = set(nodes_S_list)
        
        # Handle edge cases based on definition |E(S, V-S)| / |S|
        if n == 0 or not nodes_S or len(nodes_S) == n:
            return {"edge_expansion": 0.0}
            
        # Count edges leaving S (from nodes in S to nodes not in S)
        edges_leaving_S = 0
        for node in nodes_S:
            for neighbor in adj_list[node]:
                if neighbor not in nodes_S:
                    edges_leaving_S += 1
                    
        # Calculate edge expansion
        expansion = edges_leaving_S / len(nodes_S)
        return {"edge_expansion": float(expansion)}
        edges_leaving_S: int = 0
        for node in nodes_S:
            # For each node in S, check its neighbors
            if node < len(adj_list):  # Make sure node index is valid
                for neighbor in adj_list[node]:
                    # If neighbor is not in S, it's in V-S, so this is an edge leaving S
                    if neighbor not in nodes_S:
                        edges_leaving_S += 1

        # Calculate edge expansion
        expansion_value = edges_leaving_S / len(nodes_S)
        return {"edge_expansion": float(expansion_value)}