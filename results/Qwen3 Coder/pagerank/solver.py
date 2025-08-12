import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-06

    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> dict[str, list[float]]:
        """
        Calculates the PageRank scores for the graph using a direct NumPy implementation.
        
        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}
        
        Returns:
            A dictionary containing the PageRank scores as a list, ordered by node index.
            {"pagerank_scores": [score_node_0, score_node_1, ..., score_node_n-1]}
            Returns {"pagerank_scores": []} for n=0.
            Returns {"pagerank_scores": [1.0]} for n=1.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        # Initialize PageRank vector
        r = np.ones(n) / n
        
        # Precompute out-degrees
        out_degrees = np.array([len(neighbors) for neighbors in adj_list], dtype=float)
        
        # Identify dangling nodes (nodes with no outgoing edges)
        dangling_nodes = out_degrees == 0
        
        # Power iteration
        for _ in range(self.max_iter):
            # Compute r_new = α * P * r + (1 - α) * (1/n) * 1
            r_new = np.zeros(n)
            
            # Add contribution from dangling nodes
            dangling_sum = np.sum(r[dangling_nodes])
            r_new += self.alpha * dangling_sum / n
            
            # Add contribution from teleportation: (1 - α) * (1/n)
            r_new += (1 - self.alpha) / n
            
            # Add contribution from regular links
            for i in range(n):
                if out_degrees[i] > 0:
                    contribution = self.alpha * r[i] / out_degrees[i]
                    for j in adj_list[i]:
                        r_new[j] += contribution
            
            # Check for convergence
            if np.sum(np.abs(r_new - r)) < self.tol:
                r = r_new
                break
                
            r = r_new
        
        return {"pagerank_scores": r.tolist()}