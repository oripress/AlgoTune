import networkx as nx
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        adjacency_list = problem["adjacency_list"]
        n = len(adjacency_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        # Optimized graph construction using bulk operations
        edge_list = []
        for u, neighbors in enumerate(adjacency_list):
            for v in neighbors:
                edge_list.append((u, v))
        
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edge_list)
        
        # Compute PageRank using networkx with default parameters
        pagerank_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
        
        # Convert to list ordered by node index
        pagerank_scores = [pagerank_dict[i] for i in range(n)]
        return {"pagerank_scores": pagerank_scores}