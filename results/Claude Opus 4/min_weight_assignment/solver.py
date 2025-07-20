from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        try:
            mat = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"]
            )
        except Exception as e:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        n = mat.shape[0]
        nnz = len(problem["data"])
        density = nnz / (n * n)
        
        # Use dense solver for small/dense matrices, sparse for large/sparse
        if n <= 500 and density > 0.1:
            from scipy.optimize import linear_sum_assignment
            dense_mat = mat.toarray()
            row_ind, col_ind = linear_sum_assignment(dense_mat)
        else:
            row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)

        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}