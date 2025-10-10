from typing import Any
import numpy as np
import scipy.optimize
from sparse_utils import build_dense_from_csr

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        try:
            # Extract problem data
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            n = problem["shape"][0]
            
            # Build dense matrix with Cython
            dense_mat = build_dense_from_csr(data, indices, indptr, n)
            
            # Use linear_sum_assignment (Hungarian algorithm)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(dense_mat)
            
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
            
        except Exception as e:
            return {"assignment": {"row_ind": [], "col_ind": []}}