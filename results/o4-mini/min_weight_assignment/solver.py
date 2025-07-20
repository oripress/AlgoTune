from typing import Any
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, dict[str, list[int]]]:
        # Build CSR matrix
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = tuple(problem["shape"])
            mat = csr_matrix((data, indices, indptr), shape=shape)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Ensure square matrix
        n, m = mat.shape
        if n != m:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        num_edges = len(data)
        HUNG_THRESHOLD = 2000

        # If complete graph and size is small, use Hungarian for speed
        if num_edges == n * n and n <= HUNG_THRESHOLD:
            dense = mat.toarray()
            row_ind, col_ind = linear_sum_assignment(dense)
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}

        # Else, use sparse matching
        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(mat)
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
        except Exception:
            # Fallback to Hungarian only if complete small
            if num_edges == n * n and n <= HUNG_THRESHOLD:
                dense = mat.toarray()
                row_ind, col_ind = linear_sum_assignment(dense)
                return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
            return {"assignment": {"row_ind": [], "col_ind": []}}