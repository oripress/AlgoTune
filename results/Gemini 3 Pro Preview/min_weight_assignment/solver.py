import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem, **kwargs):
        n = problem["shape"][0]
        if n < 60:
            # For very small problems, dense solver might be faster due to overhead
            indptr = np.array(problem["indptr"])
            indices = np.array(problem["indices"])
            data = np.array(problem["data"])
            cost_matrix = np.full((n, n), np.inf)
            rows = np.repeat(np.arange(n), np.diff(indptr))
            cost_matrix[rows, indices] = data
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if np.isinf(cost_matrix[row_ind, col_ind]).any():
                    return {"assignment": {"row_ind": [], "col_ind": []}}
            return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}

        # Explicitly create numpy arrays to potentially speed up csr_matrix creation
        data = np.array(problem["data"], dtype=np.float64)
        indices = np.array(problem["indices"], dtype=np.int32)
        indptr = np.array(problem["indptr"], dtype=np.int32)
        
        mat = scipy.sparse.csr_matrix(
            (data, indices, indptr), 
            shape=problem["shape"],
            copy=False
        )
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}