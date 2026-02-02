import numpy as np
from scipy.linalg.lapack import dsyevd

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
            
        # Construct adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for i, neighbors in enumerate(adj_list):
            if neighbors:
                A[i, neighbors] = 1.0
            
        # Compute eigendecomposition using LAPACK directly
        # dsyevd computes all eigenvalues and eigenvectors of a real symmetric matrix
        # It uses a divide and conquer algorithm.
        # overwrite_a=1 allows destroying A
        w, v, info = dsyevd(A, overwrite_a=1)
        
        # Compute exp(A) = Q * exp(Lambda) * Q^T
        exp_evals = np.exp(w)
        
        # C = (v * exp_evals) @ v.T
        C = (v * exp_evals) @ v.T
        
        # Convert to dictionary
        C_list = C.tolist()
        result = {}
        for i, row in enumerate(C_list):
            result[i] = dict(enumerate(row))
            
        return {"communicability": result}