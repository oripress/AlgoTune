class Solver:
    def solve(self, problem, **kwargs):
        """
        Calculates the edge expansion for a given subset of nodes S in a directed graph.
        Uses the NetworkX definition: boundary edges in either direction divided by min(|S|, |V-S|).
        """
        adj_list = problem["adjacency_list"]
        nodes_S = problem["nodes_S"]
        n = len(adj_list)
        s_size = len(nodes_S)
        # Edge cases: empty graph, empty set S, or full set S
        if n == 0 or s_size == 0 or s_size == n:
            return {"edge_expansion": 0.0}
        # Membership mask
        mask = [False] * n
        for u in nodes_S:
            mask[u] = True
        # Count boundary edges (u->v) where u in S and v not, or u not in S and v in S
        boundary = 0
        al = adj_list
        m = mask
        for u in range(n):
            for v in al[u]:
                if m[u] != m[v]:
                    boundary += 1
        # Denominator is the smaller set size
        denom = s_size if s_size <= n - s_size else n - s_size
        return {"edge_expansion": boundary / denom}