from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from numba import njit

@njit(cache=True)
def csr_to_dense_with_big(data, indices, indptr, n, big_val):
    dense = np.full((n, n), big_val, dtype=np.float64)
    for i in range(n):
        for k in range(indptr[i], indptr[i+1]):
            dense[i, indices[k]] = data[k]
    return dense

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        try:
            n = problem["shape"][0]
            
            if n == 0:
                return {"assignment": {"row_ind": [], "col_ind": []}}
            
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            nnz = len(data)
            
            # Compute sparsity
            sparsity = nnz / (n * n) if n > 0 else 1.0
            
            # For very sparse large matrices, sparse solver is better
            if sparsity < 0.15 and n > 300:
                mat = csr_matrix((data, indices, indptr), shape=(n, n))
                row_ind, col_ind = min_weight_full_bipartite_matching(mat)
                return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
            
            # For dense/smaller matrices, use dense solver
            data_arr = np.asarray(data, dtype=np.float64)
            indices_arr = np.asarray(indices, dtype=np.int64)
            indptr_arr = np.asarray(indptr, dtype=np.int64)
            
            big_val = 1e15
            cost = csr_to_dense_with_big(data_arr, indices_arr, indptr_arr, n, big_val)
            
            row_ind, col_ind = linear_sum_assignment(cost)
            
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
        except Exception as e:
            return {"assignment": {"row_ind": [], "col_ind": []}}