import numpy as np
from scipy.sparse import coo_matrix

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-06
    
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        # Count edges and build arrays
        degrees = [len(neighbors) for neighbors in adj_list]
        m = sum(degrees)
        
        rows = np.empty(m, dtype=np.int32)
        cols = np.empty(m, dtype=np.int32) 
        data = np.empty(m, dtype=np.float64)
        dangling = np.zeros(n, dtype=np.float64)
        
        idx = 0
        for i, neighbors in enumerate(adj_list):
            d = degrees[i]
            if d == 0:
                dangling[i] = 1.0
            else:
                inv_d = 1.0 / d
                for j in neighbors:
                    rows[idx] = j
                    cols[idx] = i
                    data[idx] = inv_d
                    idx += 1
        
        P = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
        
        # Initialize PageRank vector
        r = np.full(n, 1.0 / n, dtype=np.float64)
        
        alpha = self.alpha
        teleport = (1.0 - alpha) / n
        alpha_n = alpha / n
        tol = self.tol * n
        
        # Power iteration
        for _ in range(self.max_iter):
            dangling_contrib = alpha_n * np.dot(dangling, r)
            r_new = alpha * P.dot(r)
            r_new += teleport + dangling_contrib
            
            err = np.abs(r_new - r).sum()
            if err < tol:
                return {"pagerank_scores": r_new.tolist()}
            r = r_new
        
        return {"pagerank_scores": r.tolist()}