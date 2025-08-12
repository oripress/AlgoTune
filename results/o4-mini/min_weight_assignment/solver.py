from typing import Any
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, dict[str, list[int]]]:
        """Solve the minimum weight full bipartite matching problem using sparse algorithm."""
        # Read matrix shape
        try:
            n, m = problem["shape"]
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}
        # Extract CSR data
        data = problem.get("data", [])
        indices = problem.get("indices", [])
        indptr = problem.get("indptr", [])
        # Build sparse matrix
        try:
            mat = csr_matrix((data, indices, indptr), shape=(n, m))
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}
        # Compute matching
        row_ind, col_ind = min_weight_full_bipartite_matching(mat)
        # Return as python lists
        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}