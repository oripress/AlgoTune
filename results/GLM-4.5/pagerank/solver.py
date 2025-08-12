import numpy as np
from typing import Any
import networkx as nx

class Solver:
    def __init__(self, alpha=0.85, max_iter=100, tol=1e-06):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        """
        Calculates the PageRank scores using NetworkX with highly optimized graph construction.
        
        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}
        
        Returns:
            A dictionary containing the PageRank scores as a list, ordered by node index.
            {"pagerank_scores": [score_node_0, score_node_1, ..., score_node_n-1]}
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        # Create graph directly from adjacency list using NetworkX's from_dict_of_lists
        # This is much faster than adding edges one by one
        adj_dict = {i: neighbors for i, neighbors in enumerate(adj_list)}
        G = nx.from_dict_of_lists(adj_dict, create_using=nx.DiGraph())
        
        # Calculate PageRank using networkx
        pagerank_dict = nx.pagerank(G, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        
        # Convert dict to list ordered by node index
        pagerank_list = [pagerank_dict[i] for i in range(n)]
        
        return {"pagerank_scores": pagerank_list}