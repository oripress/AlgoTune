from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment
import numba

@numba.njit
def sparse_to_dense(n, data, indices, indptr):
    """Convert sparse CSR format to dense matrix with JIT compilation."""
    dense_mat = np.full((n, n), np.inf, dtype=np.float64)
    
    for row in range(n):
        start = indptr[row]
        end = indptr[row + 1]
        for idx in range(start, end):
            col = indices[idx]
            dense_mat[row, col] = data[idx]
    
    return dense_mat

class Solver:
    def __init__(self):
        # Warm up JIT compilation
        dummy_data = np.array([1.0], dtype=np.float64)
        dummy_indices = np.array([0], dtype=np.int32)
        dummy_indptr = np.array([0, 1], dtype=np.int32)
        _ = sparse_to_dense(1, dummy_data, dummy_indices, dummy_indptr)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        """Solve the minimum weight assignment problem using scipy's linear_sum_assignment."""
        try:
            n = problem["shape"][0]
            
            # Convert to numpy arrays for Numba
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            
            # Use JIT-compiled function for matrix construction
            dense_mat = sparse_to_dense(n, data, indices, indptr)
            
            # Solve using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(dense_mat)
            
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
            
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}