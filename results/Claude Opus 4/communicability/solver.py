import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        """Calculate communicability between all pairs of nodes in a graph."""
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                A[i, j] = 1.0
        
        # Compute matrix exponential
        expA = expm(A)
        
        # Convert to dictionary format
        result = {}
        for i in range(n):
            result[i] = {}
            for j in range(n):
                result[i][j] = float(expA[i, j])
        
        return {"communicability": result}