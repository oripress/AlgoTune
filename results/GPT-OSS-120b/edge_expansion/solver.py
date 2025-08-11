from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        """
        Compute the edge expansion of a subset S in a directed graph.

        Edge expansion = |E(S, V\\S)| / |S|
        Returns 0.0 for empty S, full V, or empty graph.
        """
        adjacency = problem.get("adjacency_list", [])
        nodes_S = problem.get("nodes_S", [])
        n = len(adjacency)

        # Convert nodes_S list (unique, sorted) to a fast‑lookup boolean array
        s_len = len(nodes_S)
        membership = bytearray(n)
        for node in nodes_S:
            if 0 <= node < n:
                membership[node] = 1

        # Edge cases: empty graph, empty set, or S == V
        if n == 0 or s_len == 0 or s_len == n:
            return {"edge_expansion": 0.0}
        # Count directed edges that cross the cut (one endpoint in S, the other not)
        out_edges = 0
        adj = adjacency  # local reference
        mem = membership  # local reference
        # Iterate over all nodes to capture both outgoing and incoming crossing edges
        for u in range(n):
            u_in = mem[u]
            for v in adj[u]:
                if u_in != mem[v]:
                    out_edges += 1
        # Denominator is the smaller side of the cut, matching NetworkX's definition
        denominator = s_len if s_len <= n - s_len else n - s_len
        return {"edge_expansion": float(out_edges / denominator) if denominator else 0.0}