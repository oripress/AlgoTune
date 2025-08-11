from typing import Any, Dict, List, Set

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        """
        Compute directed edge expansion = |E(S, V-S)| / |S|.

        Input:
            problem: {
                "adjacency_list": List[List[int]],
                "nodes_S": List[int]
            }

        Returns:
            {"edge_expansion": float}
        """
        adj: List[List[int]] = problem.get("adjacency_list", []) or []
        nodes_S_list: List[int] = problem.get("nodes_S", []) or []

        n = len(adj)
        S: Set[int] = set(nodes_S_list)

        # Edge cases: empty graph, empty S, or S == V
        if n == 0 or not S:
            return {"edge_expansion": 0.0}
        if len(S) == n:
            return {"edge_expansion": 0.0}

        # Build set of nodes present in the constructed graph (0..n-1 plus any neighbors)
        graph_nodes: Set[int] = set(range(n))
        for nbrs in adj:
            for v in nbrs:
                graph_nodes.add(v)

        # If S contains nodes not present in the constructed graph, mirror reference fallback
        if not S.issubset(graph_nodes):
            return {"edge_expansion": 0.0}

        contains = S.__contains__
        cut_edges = 0

        # Count edges from u in S to v not in S.
        # Only u in [0, n-1] have adjacency lists
        for u in nodes_S_list:
            if not (0 <= u < n):
                continue
            prev = None
            for v in adj[u]:
                # adjacency lists are sorted; skip consecutive duplicates
                if v == prev:
                    continue
                prev = v
                if not contains(v):
                    cut_edges += 1

        expansion = float(cut_edges) / float(len(S))
        return {"edge_expansion": expansion}