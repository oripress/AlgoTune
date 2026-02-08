import numpy as np
import highspy
from scipy.sparse import csr_matrix

class Solver:
    def solve(self, problem, **kwargs):
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        m, n = a.shape
        
        # Compute norms of each row of a
        norms = np.linalg.norm(a, axis=1)
        
        # LP: minimize -r s.t. a @ x + norms * r <= b, r >= 0
        inf = highspy.kHighsInf
        
        h = highspy.Highs()
        h.silent()
        
        # Build constraint matrix: A_dense is m x (n+1)
        A_dense = np.empty((m, n + 1), dtype=np.float64)
        A_dense[:, :n] = a
        A_dense[:, n] = norms
        
        # Convert to CSR using scipy (vectorized, fast)
        A_csr = csr_matrix(A_dense)
        
        # Column costs, bounds
        col_cost = np.zeros(n + 1)
        col_cost[n] = -1.0
        col_lower = np.full(n + 1, -inf)
        col_lower[n] = 0.0
        col_upper = np.full(n + 1, inf)
        
        # Row bounds
        row_lower = np.full(m, -inf)
        row_upper = b.astype(np.float64)
        
        # Use passLp to set up the whole problem at once
        # passLp(num_col, num_row, num_nz, a_format, sense, offset,
        #         col_cost, col_lower, col_upper,
        #         row_lower, row_upper,
        #         a_start, a_index, a_value)
        
        # Convert to CSC for column-wise format
        A_csc = A_csr.tocsc()
        
        h.passModel(                                          # noqa
            highspy.HighsModel(                               # noqa
            )                                                  # noqa
        )                                                      # noqa
        
        # Actually, let's just add vars and rows
        h.addVars(n + 1, col_lower, col_upper)
        for j in range(n + 1):
            h.changeColCost(j, col_cost[j])                   # noqa
        
        ar_start = A_csr.indptr[:m].astype(np.int32)
        ar_index = A_csr.indices.astype(np.int32)
        ar_value = A_csr.data.astype(np.float64)
        
        h.addRows(m, row_lower, row_upper,
                  len(ar_value), ar_start, ar_index, ar_value)
        
        h.run()
        
        sol = h.getSolution()
        x_c = list(sol.col_value[:n])
        return {"solution": x_c}