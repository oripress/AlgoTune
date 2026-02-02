import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm as sparse_expm

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        # Count total edges
        total_edges = sum(len(neighbors) for neighbors in adj_list)
        
        if total_edges == 0:
            # No edges - exp of zero matrix is identity
            rng = range(n)
            result = {u: {v: (1.0 if u == v else 0.0) for v in rng} for u in rng}
            return {"communicability": result}
        
        # Determine sparsity
        density = total_edges / (n * n) if n > 0 else 0
        
        # Build COO format indices
        rows = np.empty(total_edges, dtype=np.int32)
        cols = np.empty(total_edges, dtype=np.int32)
        idx = 0
        for u, neighbors in enumerate(adj_list):
            k = len(neighbors)
            if k > 0:
                rows[idx:idx+k] = u
                cols[idx:idx+k] = neighbors
                idx += k
        
        # For reasonably sized matrices, use dense expm
        # (sparse expm returns sparse matrix which is slower for dense results)
        A = np.zeros((n, n), dtype=np.float64)
        A[rows, cols] = 1.0
        
        # Compute matrix exponential
        expA = expm(A)
        
        # Convert to dictionary format - optimized using zip
        rng = range(n)
        expA_list = expA.tolist()
        result = {u: dict(zip(rng, expA_list[u])) for u in rng}
        
        return {"communicability": result}