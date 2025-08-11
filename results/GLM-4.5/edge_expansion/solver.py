from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        """
        Calculates the edge expansion for the given subset S in the graph.
        
        This matches NetworkX's edge_expansion implementation:
        |E(S, V-S)| / min(|S|, |V-S|)

        Args:
            problem: A dictionary containing "adjacency_list" and "nodes_S".

        Returns:
            A dictionary containing the edge expansion value.
            {"edge_expansion": expansion_value}
            Returns 0.0 if S is empty or S contains all nodes.
        """
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        nodes_S = set(nodes_S_list)

        # Handle edge cases
        if n == 0 or not nodes_S:
            return {"edge_expansion": 0.0}
        if len(nodes_S) == n:
            return {"edge_expansion": 0.0}

        # Count edges from S to V-S
        edges_out = 0
        for u in nodes_S:
            for v in adj_list[u]:
                if v not in nodes_S:
                    edges_out += 1

        # NetworkX edge expansion uses min(|S|, |V-S|) as denominator
        denominator = min(len(nodes_S), n - len(nodes_S))
        expansion = edges_out / denominator

        return {"edge_expansion": expansion}