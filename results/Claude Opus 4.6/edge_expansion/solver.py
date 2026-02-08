import networkx as nx

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        s_len = len(nodes_S_list)
        
        if n == 0 or s_len == 0 or s_len == n:
            return {"edge_expansion": 0.0}
        
        # Use a set for O(1) membership testing
        s_set = set(nodes_S_list)
        
        # Count edges crossing the boundary (both directions for undirected-like behavior)
        # nx.edge_expansion counts edges in the edge boundary
        # For a DiGraph, edge_boundary(G, S) gives edges (u,v) where u in S and v not in S
        # Then expansion = |boundary| / min(|S|, |V-S|)
        
        # Let's just use networkx to be safe
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                G.add_edge(u, v)
        
        expansion = nx.edge_expansion(G, s_set)
        return {"edge_expansion": float(expansion)}