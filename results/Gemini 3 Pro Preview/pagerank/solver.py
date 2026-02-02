import networkx as nx
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
            
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                G.add_edge(u, v)
                
        alpha = getattr(self, 'alpha', 0.85)
        tol = getattr(self, 'tol', 1.0e-6)
        max_iter = getattr(self, 'max_iter', 100)
        
        try:
            pagerank_dict = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
            pagerank_list = [pagerank_dict[i] for i in range(n)]
        except:
            pagerank_list = [0.0] * n
            
        return {"pagerank_scores": pagerank_list}