import numpy as np
from scipy.sparse import csr_matrix
import itertools
import numba

@numba.njit(cache=True)
def _flatten_adj(adj_offsets, adj_flat, out_degrees, n):
    """Already done in Python; this is for the power iteration."""
    pass

@numba.njit(cache=True)
def _pagerank_power(indptr, indices, data, dangling_mask, n, alpha, tol, max_iter):
    r = np.full(n, 1.0 / n)
    r_new = np.empty(n, dtype=np.float64)
    teleport = (1.0 - alpha) / n
    
    for iteration in range(max_iter):
        # Compute dangling sum
        ds = 0.0
        for i in range(n):
            if dangling_mask[i]:
                ds += r[i]
        
        base = alpha * ds / n + teleport
        
        # Initialize r_new with base
        for i in range(n):
            r_new[i] = base
        
        # Sparse matrix-vector multiply (CSR format)
        for i in range(n):
            for k in range(indptr[i], indptr[i + 1]):
                r_new[i] += alpha * data[k] * r[indices[k]]
        
        # Check convergence
        err = 0.0
        for i in range(n):
            err += abs(r_new[i] - r[i])
        
        # Swap
        tmp = r
        r = r_new
        r_new = tmp
        
        if err < n * tol:
            break
    
    return r

class Solver:
    def __init__(self):
        # Trigger JIT compilation with a small problem
        indptr = np.array([0, 1, 2], dtype=np.int32)
        indices = np.array([1, 0], dtype=np.int32)
        data = np.array([1.0, 1.0], dtype=np.float64)
        dm = np.array([False, False])
        _pagerank_power(indptr, indices, data, dm, 2, 0.85, 1e-6, 1)

    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        alpha = 0.85
        tol = 1.0e-06
        max_iter = 100
        
        # Compute out-degrees
        out_degrees = np.array([len(nb) for nb in adj_list], dtype=np.int32)
        total_edges = int(out_degrees.sum())
        
        if total_edges == 0:
            return {"pagerank_scores": [1.0 / n] * n}
        
        # Build sparse transition matrix
        rows = np.fromiter(
            itertools.chain.from_iterable(adj_list),
            dtype=np.int32,
            count=total_edges
        )
        cols = np.repeat(np.arange(n, dtype=np.int32), out_degrees)
        
        inv_deg = np.zeros(n, dtype=np.float64)
        nz = out_degrees > 0
        inv_deg[nz] = 1.0 / out_degrees[nz]
        data = inv_deg[cols]
        
        M = csr_matrix((data, (rows, cols)), shape=(n, n))
        
        dangling_mask = ~nz
        
        r = _pagerank_power(
            M.indptr.astype(np.int32),
            M.indices.astype(np.int32),
            M.data,
            dangling_mask,
            n, alpha, tol, max_iter
        )
        
        return {"pagerank_scores": r.tolist()}