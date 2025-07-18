import networkx as nx
from networkx.algorithms.clique import max_weight_clique

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        
        # Create graph and add edges efficiently
        G = nx.Graph()
        
        # Add edges using generator expression for better performance
        G.add_edges_from(
            (i, j)
            for i in range(n)
            for j in range(i+1, n)
            if problem[i][j] == 1
        )
        
        # Find maximum clique - use weight=None for uniform weights
        clique_nodes, _ = max_weight_clique(G, weight=None)
        return list(clique_nodes)