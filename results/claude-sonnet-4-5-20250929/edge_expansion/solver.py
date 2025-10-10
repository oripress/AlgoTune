from typing import Any
import networkx as nx

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        """
        Calculates the edge expansion for the given subset S in the graph using NetworkX.
        """
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        nodes_S: set[int] = set(nodes_S_list)
        
        # Handle edge cases
        if n == 0 or not nodes_S:
            return {"edge_expansion": 0.0}
        if len(nodes_S) == n:
            return {"edge_expansion": 0.0}
        
        # Reconstruct the NetworkX DiGraph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                G.add_edge(u, v)
        
        # Calculate edge expansion using networkx
        try:
            expansion = nx.edge_expansion(G, nodes_S)
            expansion_value = float(expansion)
        except Exception as e:
            expansion_value = 0.0
        
        solution = {"edge_expansion": expansion_value}
        return solution