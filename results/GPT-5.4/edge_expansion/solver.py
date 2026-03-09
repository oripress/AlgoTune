from typing import Any

import networkx as nx

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        nodes_s_list = problem["nodes_S"]
        n = len(adj_list)
        nodes_s = set(nodes_s_list)

        if n == 0 or not nodes_s:
            return {"edge_expansion": 0.0}
        if len(nodes_s) == n:
            return {"edge_expansion": 0.0}

        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                g.add_edge(u, v)

        try:
            expansion_value = float(nx.edge_expansion(g, nodes_s))
        except Exception:
            expansion_value = 0.0

        return {"edge_expansion": expansion_value}