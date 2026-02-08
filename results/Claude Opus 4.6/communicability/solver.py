import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        if n == 1:
            return {"communicability": {0: {0: 1.0}}}
        
        # Build adjacency matrix using vectorized approach
        total = 0
        for neighbors in adj_list:
            total += len(neighbors)
        
        if total > 0:
            rows = np.empty(total, dtype=np.intp)
            cols = np.empty(total, dtype=np.intp)
            idx = 0
            for i, neighbors in enumerate(adj_list):
                nn = len(neighbors)
                if nn > 0:
                    rows[idx:idx+nn] = i
                    cols[idx:idx+nn] = neighbors
                    idx += nn
            
            A = np.zeros((n, n), dtype=np.float64)
            A[rows, cols] = 1.0
        else:
            A = np.zeros((n, n), dtype=np.float64)
        
        # Use eigendecomposition for symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        exp_eigenvalues = np.exp(eigenvalues)
        # exp(A) = V * diag(exp(lambda)) * V^T
        result_matrix = (eigenvectors * exp_eigenvalues) @ eigenvectors.T
        
        # Convert to dict of dicts efficiently
        result_comm_dict = {}
        for u in range(n):
            result_comm_dict[u] = dict(enumerate(result_matrix[u].tolist()))
        
        return {"communicability": result_comm_dict}