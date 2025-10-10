import numpy as np
from scipy.sparse import csr_matrix

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-6

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        """
        Calculates PageRank scores using optimized power iteration.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build sparse transition matrix P with pre-allocated arrays
        row_indices = []
        col_indices = []
        data = []
        dangling_nodes = []
        
        for i in range(n):
            neighbors = adj_list[i]
            deg = len(neighbors)
            if deg > 0:
                weight = 1.0 / deg
                for j in neighbors:
                    row_indices.append(j)
                    col_indices.append(i)
                    data.append(weight)
            else:
                dangling_nodes.append(i)
        
        P = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=np.float64)
        
        # Initialize PageRank scores uniformly
        r = np.full(n, 1.0 / n, dtype=np.float64)
        teleport = (1.0 - self.alpha) / n
        alpha_over_n = self.alpha / n
        threshold = n * self.tol
        has_dangling = len(dangling_nodes) > 0
        
        # Power iteration
        for _ in range(self.max_iter):
            r_new = P.dot(r) * self.alpha
            
            # Handle dangling nodes
            if has_dangling:
                dangling_sum = sum(r[i] for i in dangling_nodes)
                r_new += alpha_over_n * dangling_sum
            
            # Add teleportation
            r_new += teleport
            
            # Check convergence using L1 norm
            if np.abs(r_new - r).sum() < threshold:
                return {"pagerank_scores": r_new.tolist()}
            
            r = r_new
        
        return {"pagerank_scores": r.tolist()}