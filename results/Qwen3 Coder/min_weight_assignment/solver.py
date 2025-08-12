from typing import Any
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import scipy.sparse

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """Solve the minimum weight assignment problem."""
        try:
            # Direct matrix creation and solving with minimal overhead
            mat = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), 
                shape=problem["shape"]
            )
            row_ind, col_ind = min_weight_full_bipartite_matching(mat)
            return {"assignment": {"row_ind": row_ind, "col_ind": col_ind}}
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}