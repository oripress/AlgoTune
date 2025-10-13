from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        """
        Compute the edge expansion of subset S in a directed graph G=(V,E).

        NetworkX's edge_expansion is used as the reference; it computes:
            edge_expansion(S) = |E(S, V-S)| / min(|S|, |V-S|)
        where E(S, V-S) are directed edges from S to its complement.

        Args:
            problem: A dictionary with:
                - "adjacency_list": List[List[int]] adjacency list of the directed graph.
                - "nodes_S": List[int] sorted list of node indices in S.

        Returns:
            Dict[str, float]: {"edge_expansion": float}
        """
        adj_list: List[List[int]] = problem["adjacency_list"]
        nodes_S_list: List[int] = problem["nodes_S"]

        n = len(adj_list)
        s_size = len(nodes_S_list)

        # Edge cases aligned with validator expectations
        if n == 0 or s_size == 0 or s_size == n:
            return {"edge_expansion": 0.0}

        S_set = set(nodes_S_list)

        # Determine total number of nodes as NetworkX would after adding edges:
        # start with nodes 0..n-1, then include any neighbor nodes (can be outside [0, n-1])
        extras = set()
        for neighbors in adj_list:
            # neighbors are sorted per problem statement; duplicates are fine for node counting
            for v in neighbors:
                if v < 0 or v >= n:
                    extras.add(v)
        total_nodes = n + len(extras)

        # If S contains a node not present in the constructed graph, NetworkX would error.
        # The reference catches exceptions and returns 0.0, so we mirror that behavior.
        for u in nodes_S_list:
            if u < 0 or u >= n:
                if u not in extras:
                    return {"edge_expansion": 0.0}

        # Denominator uses min(|S|, |V-S|) with V size equal to total_nodes
        denom = min(s_size, total_nodes - s_size)
        if denom <= 0:
            # Matches the reference behavior (would raise in NX and be caught -> 0.0)
            return {"edge_expansion": 0.0}

        # Count unique directed edges from S to V-S.
        # Deduplicate parallel entries in adjacency (NetworkX DiGraph collapses duplicates).
        cut_edges = 0
        S = S_set  # local alias

        for u in nodes_S_list:
            if 0 <= u < n:
                prev = None  # since neighbors are sorted, prev deduplicates identical consecutive v
                for v in adj_list[u]:
                    if v != prev:
                        if v not in S:
                            cut_edges += 1
                        prev = v

        expansion_value = float(cut_edges) / float(denom)
        return {"edge_expansion": expansion_value}