import numpy as np
from scipy import sparse
from typing import Any

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-6
    
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[float]]:
        """
        Fast PageRank implementation using sparse matrices.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        # Handle edge cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        # Build sparse transition matrix
        row_ind = []
        col_ind = []
        data = []
        
        dangling_nodes = []
        
        for i in range(n):
            out_degree = len(adj_list[i])
            if out_degree > 0:
                prob = 1.0 / out_degree
                for j in adj_list[i]:
                    row_ind.append(j)
                    col_ind.append(i)
                    data.append(prob)
            else:
                dangling_nodes.append(i)
        
        # Create sparse matrix (CSC format is efficient for matrix-vector multiplication)
        if len(data) > 0:
            P = sparse.csc_matrix((data, (row_ind, col_ind)), shape=(n, n), dtype=np.float64)
        else:
            P = sparse.csc_matrix((n, n), dtype=np.float64)
        
        # Initialize PageRank vector
        r = np.ones(n, dtype=np.float64) / n
        
        # Precompute teleportation vector
        teleport = (1 - self.alpha) / n
        
        # Convert dangling nodes to boolean mask for faster access
        is_dangling = np.zeros(n, dtype=bool)
        is_dangling[dangling_nodes] = True
        
        # Power iteration
        for _ in range(self.max_iter):
            r_old = r
            
            # Compute dangling node contribution
            dangling_sum = np.sum(r_old[is_dangling])
            
            # New PageRank: r = alpha * P * r_old + (teleport + alpha * dangling_sum / n)
            r = self.alpha * (P @ r_old) + (teleport + self.alpha * dangling_sum / n)
            
            # Check convergence
            if np.sum(np.abs(r - r_old)) < n * self.tol:
                break
        
        # Normalize to ensure sum is exactly 1.0
        r = r / np.sum(r)
        
        return {"pagerank_scores": r.tolist()}