from typing import Any
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, dict[str, list[int]]]:
        import numpy as np
        mat = scipy.sparse.csr_matrix.__new__(scipy.sparse.csr_matrix)
        mat.data = np.array(problem["data"], dtype=np.float64)
        mat.indices = np.array(problem["indices"], dtype=np.int32)
        mat.indptr = np.array(problem["indptr"], dtype=np.int32)
        mat._shape = tuple(problem["shape"])
        
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
        return {"assignment": {"row_ind": row_ind, "col_ind": col_ind}}