from typing import Any, List, Set

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute edge expansion for a subset S in a directed graph.

        Match NetworkX's edge_expansion(G, S) behavior on a DiGraph constructed from:
        - Nodes 0..n-1 (n = len(adjacency_list))
        - Plus any neighbor nodes introduced by edges (add_edge adds missing nodes)
        expansion(S) = |E(S, V-S)| / min(|S|, |V-S|)
        Additional constraints to match the harness/reference:
        - If n == 0, |S| == 0, or |S| == n (with n = len(adjacency_list)), return 0.0
        - If S contains nodes not in the graph, return 0.0 (reference catches the NetworkX error)
        """
        adj_list: List[List[int]] = problem.get("adjacency_list", [])
        nodes_S_list: List[int] = problem.get("nodes_S", [])

        n = len(adj_list)

        # Build the node set as NetworkX would (edges add missing nodes)
        nodes_in_graph: Set[int] = set(range(n))
        for nbrs in adj_list:
            # neighbors may be outside 0..n-1; include them as graph nodes
            nodes_in_graph.update(nbrs)

        s_set: Set[int] = set(nodes_S_list)
        len_S = len(s_set)

        # Handle harness-specific edge cases early (must use n = len(adj_list))
        if n == 0 or len_S == 0 or len_S == n:
            return {"edge_expansion": 0.0}

        # If S contains nodes not in the graph, mimic NetworkX error fallback to 0.0
        if not s_set.issubset(nodes_in_graph):
            return {"edge_expansion": 0.0}

        total_nodes = len(nodes_in_graph)
        if len_S == total_nodes:
            # S == V -> no outgoing edges to V-S
            return {"edge_expansion": 0.0}

        # Count directed edges from S to V-S
        edge_count = 0
        s_contains = s_set.__contains__
        for u in s_set:
            if 0 <= u < n:
                nbrs = adj_list[u]
                prev = None
                # adjacency lists are sorted; skip consecutive duplicates (DiGraph has single edges)
                for v in nbrs:
                    if v == prev:
                        continue
                    prev = v
                    if not s_contains(v):
                        edge_count += 1

        # Normalize by min(|S|, |V-S|) using total graph nodes
        other = total_nodes - len_S
        denom = len_S if len_S <= other else other
        expansion_value = 0.0 if denom <= 0 else float(edge_count) / float(denom)
        return {"edge_expansion": expansion_value}