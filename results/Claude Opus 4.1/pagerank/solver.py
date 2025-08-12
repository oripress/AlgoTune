import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        # Default parameters matching NetworkX defaults
        self.alpha = 0.85
        self.tol = 1e-6
        self.max_iter = 100
    
    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> dict[str, list[float]]:
        """
        Optimized PageRank implementation using NumPy.
        Matches NetworkX's power iteration method.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        # Edge cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        # Count dangling nodes
        dangling = []
        for i in range(n):
            if len(adj_list[i]) == 0:
                dangling.append(i)
        
        # Initialize PageRank vector uniformly
        r = np.ones(n, dtype=np.float64) / n
        
        # Precompute values for efficiency
        personalization = np.ones(n, dtype=np.float64) / n
        
        # Power iteration
        for iteration in range(self.max_iter):
            r_prev = r.copy()
            r = np.zeros(n, dtype=np.float64)
            
            # Distribute rank from non-dangling nodes
            for i in range(n):
                if len(adj_list[i]) > 0:
                    out_degree = len(adj_list[i])
                    for j in adj_list[i]:
                        r[j] += self.alpha * r_prev[i] / out_degree
            
            # Handle dangling nodes
            dangling_sum = sum(r_prev[d] for d in dangling)
            r += self.alpha * dangling_sum / n
            
            # Add teleportation
            r += (1 - self.alpha) * personalization
            
            # Normalize to handle numerical errors
            r = r / r.sum()
            
            # Check convergence
            if np.sum(np.abs(r - r_prev)) < self.tol:
                break
        
        # Convert to list and return
        return {"pagerank_scores": r.tolist()}