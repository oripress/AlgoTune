from typing import Any
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        This is the reference implementation to be improved.
        """
        try:
            mat = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"]
            )
        except (ValueError, IndexError):
            return {"assignment": {"row_ind": [], "col_ind": []}}

        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(mat)
        except ValueError:
            # This can happen if a perfect matching is not possible.
            return {"assignment": {"row_ind": [], "col_ind": []}}

        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}