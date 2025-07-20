import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from numba import njit
from typing import Any

@njit
def precompute_missing_value(data, n):
    min_val = np.inf
    max_val = -np.inf
    for i in range(len(data)):
        x = data[i]
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x
    return (max_val - min_val) * (n + 1) + max_val + 1

@njit
def build_cost_matrix(n, data, indices, indptr, missing_value):
    cost_matrix = np.full((n, n), missing_value, dtype=np.float32)
    for i in range(n):
        start = indptr[i]
        end = indptr[i+1]
        for idx in range(start, end):
            j = indices[idx]
            cost_matrix[i, j] = np.float32(data[idx])
    return cost_matrix

class Solver:
    def solve(self, problem, **kwargs):
        try:
            n = problem["shape"][0]
            if n == 0:
                return {"assignment": {"row_ind": [], "col_ind": []}}
            
            # Use optimized cost matrix construction
            data = np.array(problem["data"], dtype=np.float32)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            
            # Handle invalid indptr length
            if len(indptr) < n + 1:
                return {"assignment": {"row_ind": [], "col_ind": []}}
            
            # Precompute missing value
            missing_value = precompute_missing_value(data, n)
            
            # Build cost matrix with JIT optimization
            cost_matrix = build_cost_matrix(n, data, indices, indptr, missing_value)
            
            # Apply Hungarian algorithm
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}