import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from typing import Any, Dict
class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the communicability for all pairs of nodes in a graph using scipy's expm."""
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        # Build adjacency matrix using vectorized operations
        A = np.zeros((n, n), dtype=float)
        rows = np.repeat(np.arange(n, dtype=int), [len(neighbors) for neighbors in adj_list])
        cols = np.concatenate(adj_list).astype(int)
        A[rows, cols] = 1
        A[cols, rows] = 1  # Since it's undirected
        # Compute matrix exponential using scipy's optimized expm
        exp_A = expm(A)
        
        # Convert to dictionary format using numpy's array methods
        communicability = {}
        for i in range(n):
            communicability[i] = dict(enumerate(exp_A[i].tolist()))
        
        return {"communicability": communicability}