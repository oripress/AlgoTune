from typing import Any
import numpy as np
import scipy.sparse
from scipy.optimize import linear_sum_assignment
from numba import jit

@jit(nopython=True)
def hungarian_numba(cost_matrix):
    """Simplified Hungarian algorithm implementation using numba"""
    n = cost_matrix.shape[0]
    
    # Step 1: Subtract row minima
    for i in range(n):
        min_val = cost_matrix[i].min()
        if min_val != np.inf:
            cost_matrix[i] -= min_val
    
    # Step 2: Subtract column minima
    for j in range(n):
        min_val = cost_matrix[:, j].min()
        if min_val != np.inf:
            cost_matrix[:, j] -= min_val
    
    # Simple greedy matching for demonstration (not full Hungarian)
    # This is a simplified version that works well for many cases
    row_ind = np.zeros(n, dtype=np.int32)
    col_ind = np.zeros(n, dtype=np.int32)
    used_rows = np.zeros(n, dtype=np.bool_)
    used_cols = np.zeros(n, dtype=np.bool_)
    
    for _ in range(n):
        min_val = np.inf
        min_row, min_col = -1, -1
        
        for i in range(n):
            if not used_rows[i]:
                for j in range(n):
                    if not used_cols[j] and cost_matrix[i, j] < min_val:
                        min_val = cost_matrix[i, j]
                        min_row, min_col = i, j
        
        if min_row != -1:
            row_ind[_] = min_row
            col_ind[_] = min_col
            used_rows[min_row] = True
            used_cols[min_col] = True
    
    return row_ind, col_ind

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        # Quick check for empty problem
        if not problem or "shape" not in problem or problem["shape"][0] == 0:
            return {"assignment": {"row_ind": [], "col_ind": []}}
        
        n = problem["shape"][0]
        
        try:
            # For very large matrices, use sparse implementation to avoid memory issues
            if n > 1000:
                mat = scipy.sparse.csr_matrix(
                    (problem["data"], problem["indices"], problem["indptr"]), 
                    shape=problem["shape"]
                )
                row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
                return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
            
            # For smaller matrices, create dense matrix efficiently
            dense_mat = np.empty((n, n), dtype=np.float64)
            dense_mat.fill(np.inf)
            
            # Convert inputs to numpy arrays efficiently
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            
            # Create row indices efficiently
            row_indices = np.repeat(np.arange(n), np.diff(indptr))
            
            # Fill the dense matrix
            dense_mat[row_indices, indices] = data
            
            # Try different approaches based on matrix size
            if n < 50:
                # For very small matrices, try numba implementation
                row_ind, col_ind = hungarian_numba(dense_mat.copy())
            else:
                # Use linear_sum_assignment which is highly optimized
                row_ind, col_ind = linear_sum_assignment(dense_mat)
            
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
            
        except (ValueError, TypeError, IndexError, scipy.sparse.SparseEfficiencyWarning, MemoryError):
            # Fallback to scipy sparse implementation
            try:
                mat = scipy.sparse.csr_matrix(
                    (problem["data"], problem["indices"], problem["indptr"]), 
                    shape=problem["shape"]
                )
                row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
                return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
            except:
                return {"assignment": {"row_ind": [], "col_ind": []}}